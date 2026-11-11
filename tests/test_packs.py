"""Evaluation packs and CPU rescoring.

The first group needs no data.  The round-trip test needs the project outputs
(``MLIPEVAL_OUTPUT``) and a reference pack (``MLIPEVAL_REFERENCE``); it is
skipped without them::

    MLIPEVAL_OUTPUT=.../eval_framework MLIPEVAL_REFERENCE=.../gst225_phase1_v1.mlipref \\
        python -m pytest tests/test_packs.py
"""

from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval_framework" / "scripts"))

from mlipeval import l1_pes, l2_static, pack  # noqa: E402
from mlipeval.schema import Status  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MLIPEVAL_PACK_CACHE", str(tmp_path / "unpacked"))


# ------------------------------------------------------------------ container
def _stage(tmp_path: Path) -> Path:
    stage = tmp_path / "stage"
    (stage / "L1").mkdir(parents=True)
    (stage / "L1" / "probes.json").write_text('{"a": 1}')
    np.savez(stage / "L1" / "x.npz", v=np.arange(3))
    return stage


def test_container_roundtrip(tmp_path):
    out = pack.write_container(_stage(tmp_path), tmp_path / "p.zip", {"schema": "t", "kind": "test"})
    root = pack.extract(out)
    m = pack.read_manifest(root)
    assert set(m["files"]) == {"L1/probes.json", "L1/x.npz"}
    assert pack.verify_files(root, m) == []
    assert pack.extract(out) == root  # idempotent


def test_tampered_member_is_rejected(tmp_path):
    out = pack.write_container(_stage(tmp_path), tmp_path / "p.zip", {"schema": "t", "kind": "test"})
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(out) as src, zipfile.ZipFile(bad, "w") as dst:
        for info in src.infolist():
            data = src.read(info.filename)
            dst.writestr(info.filename, b'{"a": 2}' if info.filename == "L1/probes.json" else data)
    with pytest.raises(pack.PackError, match="sha256 mismatch"):
        pack.extract(bad, tmp_path / "x")


def test_unsafe_member_is_rejected(tmp_path):
    bad = tmp_path / "slip.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"kind": "test", "files": {}}))
        zf.writestr("../evil.txt", "x")
    with pytest.raises(pack.PackError, match="unsafe member"):
        pack.extract(bad, tmp_path / "x")
    assert not (tmp_path / "evil.txt").exists()


# --------------------------------------------------------------- L1 judges
def _probe_results(noise=1e-12):
    return (
        l1_pes.invariance_metrics(d_translation=2e-7, d_rotation=3e-6, net_force=1e-9, net_torque=0.1,
                                  n_atoms=306, noise_eV=noise)
        + [l1_pes.force_consistency_metric(rel=3e-6, delta=1e-3, n_probed=8, scale=0.9, noise_eV=noise),
           l1_pes.stress_consistency_metric(analytic=[1e-3, 2e-3, 1e-3, 0, 0, 1e-5],
                                            numeric=[1e-3, 2e-3, 1e-3, 0, 0, 1.1e-5],
                                            eps=1e-5, volume=9900.0, noise_eV=noise)]
    )


def test_probe_record_rejudges_identically():
    live = _probe_results()
    record = l1_pes.probe_record(live, n_atoms=306, volume_A3=9900.0, noise_eV=1e-12)
    again = l1_pes.probe_metrics(record)
    assert [r.to_dict() for r in again] == [r.to_dict() for r in live]
    # JSON form, as a result file stores it
    record_json = l1_pes.probe_record([r.to_dict() for r in live], n_atoms=306, volume_A3=9900.0,
                                      noise_eV=1e-12)
    assert record_json == record


def test_noise_floor_widens_tolerance():
    # 5 ueV breaks the 1 ueV float64 tolerance, but sits inside 5x a 10 ueV float32 noise floor
    tight = l1_pes.invariance_metrics(d_translation=5e-6, d_rotation=0, net_force=0, net_torque=0,
                                      n_atoms=306, noise_eV=0.0)[0]
    loose = l1_pes.invariance_metrics(d_translation=5e-6, d_rotation=0, net_force=0, net_torque=0,
                                      n_atoms=306, noise_eV=1e-5)[0]
    assert tight.status is Status.FAILED and loose.status is Status.OK


# --------------------------------------------------------------- L2 records
def test_relaxation_record_matches_live():
    from ase.build import bulk

    atoms = bulk("Ge", "diamond", a=5.7)
    relaxed = atoms.copy()
    relaxed.set_cell(atoms.cell.array * 1.01, scale_atoms=True)
    rr = l2_static.RelaxResult(
        label="element_Ge", initial=atoms, relaxed=relaxed, energy=-9.0, energy_per_atom=-4.5,
        n_steps=12, fmax=0.01, max_force=0.004, converged=True,
        volume_change_pct=3.03, cell_lengths=np.asarray(relaxed.cell.lengths()),
        cell_angles=np.asarray(relaxed.cell.angles()))
    dft = {"volume_per_atom_A3": 23.0, "cell_lengths_A": [4.0, 4.0, 4.0]}
    live = [r.to_dict() for r in l2_static.relaxation_metrics(rr, dft_reference=dft)]
    stored = json.loads(json.dumps(rr.record()))
    again = [r.to_dict() for r in l2_static.relaxation_record_metrics(stored, dft_reference=dft)]
    assert again == live
    assert stored["composition"] == {"Ge": 2}


class _FakeModel:
    """Enough of mlipeval.calculators.Model for mlip_eos: an ASE calculator and a name."""

    def __init__(self):
        from ase.calculators.lj import LennardJones

        self.calculator = LennardJones(sigma=2.3, epsilon=0.1, rc=6.0)

        class _Spec:
            name = "lj"

        self.spec = _Spec()


def test_mlip_eos_scans_volume_like_the_dft_tasks():
    from ase.build import bulk

    atoms = bulk("Ar", "fcc", a=5.26)
    curve = l2_static.mlip_eos(_FakeModel(), atoms, label="t", relax_ions=False)
    ratio = curve.volumes / atoms.get_volume()
    assert np.allclose(ratio, [0.92, 0.95, 0.98, 1.00, 1.02, 1.05, 1.08])
    legacy = l2_static.mlip_eos(_FakeModel(), atoms, label="t", relax_ions=False, scale_kind="linear")
    dft = l2_static.EOSCurve("d", atoms.get_volume() * np.array([0.92, 0.95, 0.98, 1, 1.02, 1.05, 1.08]),
                             np.zeros(7), len(atoms))
    assert l2_static.eos_window(curve, dft)["matched"]
    assert not l2_static.eos_window(legacy, dft)["matched"]


# ------------------------------------------------------- round trip on data
REFERENCE = os.environ.get("MLIPEVAL_REFERENCE")
OUTPUT = os.environ.get("MLIPEVAL_OUTPUT")


@pytest.mark.skipif(not (REFERENCE and OUTPUT and Path(REFERENCE).exists()),
                    reason="needs MLIPEVAL_OUTPUT and MLIPEVAL_REFERENCE")
@pytest.mark.parametrize("model_key", ["mace_mpa0", "nequip_spec_gst"])
def test_pack_rescore_reproduces_hpc_results(tmp_path, model_key):
    from mlipeval import rescore

    ref = pack.open_reference(REFERENCE)
    path, _summary = pack.build_model_pack(model_key, tmp_path / f"{model_key}.mlippack", reference=ref)
    out = rescore.rescore(pack.open_model_pack(path), ref)
    assert out.verification["ok"]
    cmp = rescore.compare_with_pack(out)
    for stem, c in cmp.items():
        assert c["identical"], (stem, c["diffs"][:3])
    written = out.save(tmp_path / "rescored")
    assert {"L0_audit", "L1", "L2"} <= set(written)
