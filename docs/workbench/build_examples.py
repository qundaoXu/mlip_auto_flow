#!/usr/bin/env python
"""Export the GST225 demo reference and the nine showcase models as upload files.

    MLIPEVAL_OUTPUT=/path/to/eval_framework python docs/workbench/build_examples.py

Writes ``docs/workbench/examples/gst225/`` in exactly the formats the Workbench's
*New evaluation* panel accepts from a user, so the demo goes through the same
parsers and the same in-browser analysis as a real upload:

    reference/  l0_labels.extxyz.gz   DFT energies, forces, stresses of the 120 H5 audit frames
                l2_reference.json     DFT relaxations/statics, compositions, phase-order pair
                l2_eos.csv            DFT fixed-volume EOS points
    models/<key>/
                l0_predictions.extxyz.gz   model energies, forces, stresses on the same frames
                l1_pec.csv, l1_lattice_scan.csv, l1_nve_dt3.csv, l1_nve_dt1.5.csv,
                l1_stability_2000K.csv, l1_probes.json
                l2_relax.csv, l2_eos.csv
    expected_metrics.json   the HPC values of every scalar L0/L1/L2 metric, used to
                            show (and test) that the browser reproduces them

Reads the reference pack (``data/reference/gst225_phase1_v1.mlipref``) and the
per-model outputs under ``MLIPEVAL_OUTPUT``.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "eval_framework" / "scripts"))
if not os.environ.get("MLIPEVAL_OUTPUT"):
    sys.exit("set MLIPEVAL_OUTPUT to the eval_framework output tree (results/, data/)")

from mlipeval import pack, paths  # noqa: E402
from mlipeval.cache import frames_fingerprint, load_predictions  # noqa: E402
from mlipeval.l1_pes import NOISE_MARGIN, probe_record  # noqa: E402

SHOW = ["mace_mpa0", "nequip_oam_l", "dpa4_pro", "mattersim_5m", "chgnet_030",
        "mace_spec_gst", "mace_spec_gst_ne", "nequip_spec_gst", "mace_gst_aging"]
OUT = Path(__file__).resolve().parent / "examples" / "gst225"
F = "{:.10g}".format   # coordinates and L0 labels
G = "{:.17g}".format   # L1/L2 tables: drifts and EOS fits need full double precision


def gz_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0: identical content gives identical bytes, so rebuilds do not churn git
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, compresslevel=9) as gz:
        gz.write(text.encode())


def write_csv(path: Path, header: list[str], rows, comments: list[str] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    for c in comments:
        buf.write(f"# {c}\n")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for row in rows:
        w.writerow([G(v) if isinstance(v, float) else v for v in row])
    path.write_text(buf.getvalue())


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")


def lattice(cell) -> str:
    return " ".join(F(float(v)) for v in np.asarray(cell).reshape(-1))


def flat9(m) -> str:
    return " ".join(F(float(v)) for v in np.asarray(m).reshape(-1))


# ------------------------------------------------------------------ reference
ref = pack.open_reference(pack.default_reference_path())
h5 = ref.h5()
fp = frames_fingerprint(h5.frames)
lines = []
for a in h5.frames:
    info = a.info
    lines.append(str(len(a)))
    lines.append(
        f'Lattice="{lattice(a.cell.array)}" Properties=species:S:1:pos:R:3:REF_forces:R:3 '
        f'pbc="T T T" REF_energy={F(float(info["REF_energy"]))} '
        f'REF_stress="{flat9(info["REF_stress"])}" '
        f'pool={info["pool"]} state_kind={info["state_kind"]} config_type={info["config_type"]}')
    for s, p, f in zip(a.get_chemical_symbols(), a.positions, a.arrays["REF_forces"]):
        lines.append(f"{s} {F(p[0])} {F(p[1])} {F(p[2])} {F(f[0])} {F(f[1])} {F(f[2])}")
gz_write(OUT / "reference" / "l0_labels.extxyz.gz", "\n".join(lines) + "\n")

l2 = ref.l2()
tasks = l2["tasks"]
structures = []
ladder = {row[0]: row for row in l2["phase_ladder"]}
seed = 1


def dft_row(key, role, relax_label, name, static_task, relax_task, composition):
    st, rx = tasks[static_task], tasks[relax_task]
    return {"key": key, "role": role, "relax_label": relax_label, "name": name,
            "composition": composition, "energy_eV_per_atom": st["energy_per_atom_eV"],
            "volume_A3_per_atom": rx["volume_per_atom_A3"], "cell_lengths_A": rx["cell_lengths_A"],
            "static_task": static_task, "relax_task": relax_task}


for key, source, static, label in l2["phase_ladder"]:
    if source == "cubic_seed":
        structures.append(dft_row("cubic", "phase", f"cubic_s{seed:02d}", label,
                                  static.format(seed=seed), static.format(seed=seed).replace("_static", "_relax"),
                                  l2["compositions"][f"cubic_s{seed:02d}"]))
    else:
        relax_label = "hex_mp1224375" if key == "mp1224375" else f"phase_{key}"
        structures.append(dft_row(key, "phase", relax_label, label, static, static.replace("_static", "_relax"),
                                  l2["compositions"][key]))
for el in l2["elements"]:
    structures.append(dft_row(el, "element", f"element_{el}", f"{el} ground state",
                              f"L2_{el.lower()}_static", f"L2_{el.lower()}_relax", l2["compositions"][el]))
write_json(OUT / "reference" / "l2_reference.json", {
    "format": "mlipeval-l2-reference/1",
    "system": "GST225 (Ge2Sb2Te5)",
    "protocol_id": l2["dft_protocol"],
    "structures": structures,
    "phase_order": {"phase": "cubic", "base": "mp1224375", "names": ["cubic", "hexagonal"]},
    "eos_groups": {"hex": "Hexagonal", "cubic": "Cubic (five vacancy orderings)"},
})
eos_rows = []
for phase, rec in l2["eos"].items():
    group, member = ("hex", "") if phase == "hex" else ("cubic", f"s{phase[3:]}")
    n = rec["n_atoms"]
    for v, e in zip(rec["volumes_A3"], rec["energies_eV"]):
        eos_rows.append([group, member, v / n, e / n])
write_json(OUT / "reference" / "l1_reference.json", {
    "format": "mlipeval-l1-reference/1",
    "aimd_nve_drift_meV_per_atom_per_ps": ref.l1()["aimd_nve_reference"]["drift_meV_per_atom_per_ps"],
    "note": "AIMD NVE run of the same cell, density, temperature and timestep (H0_preflight_03_nve1200); "
            "optional -- with it the NVE drift check is a like-for-like comparison",
})
write_csv(OUT / "reference" / "l2_eos.csv", ["group", "member", "volume_A3_per_atom", "energy_eV_per_atom"],
          eos_rows, ["DFT fixed-volume, ion-relaxed EOS (gst225_pbe_canonical_v1)"])

# ------------------------------------------------------------------ models
expected = {}
for key in SHOW:
    out = OUT / "models" / key
    res_dir = paths.MODEL_RESULTS_ROOT / key
    data_dir = paths.MODEL_DATA_ROOT / key
    audit = json.loads((res_dir / "L0_audit.json").read_text())
    spec = audit["model"]

    # L0 predictions on the same frames
    pred = load_predictions(pack._find_cache(paths.CACHE, f"L0_h5_{len(h5)}", spec, fp))
    lines = []
    for i, a in enumerate(h5.frames):
        lines.append(str(len(a)))
        stress = f' MLIP_stress="{flat9(pred.stresses[i])}"' if pred.stresses is not None else ""
        lines.append(f'Lattice="{lattice(a.cell.array)}" Properties=species:S:1:pos:R:3:MLIP_forces:R:3 '
                     f'pbc="T T T" MLIP_energy={F(float(pred.energies[i]))}{stress} '
                     f'config_type={a.info["config_type"]}')
        for s, p, f in zip(a.get_chemical_symbols(), a.positions, pred.forces[i]):
            lines.append(f"{s} {F(p[0])} {F(p[1])} {F(p[2])} {F(f[0])} {F(f[1])} {F(f[2])}")
    gz_write(out / "l0_predictions.extxyz.gz", "\n".join(lines) + "\n")

    # L1
    l1_json = json.loads(next(res_dir.glob("L1_*.json")).read_text())
    z = np.load(pack._find_pec(paths.CACHE, spec), allow_pickle=False)
    r = z["r"]
    rows, noise = [], {}
    for i, pair in enumerate(str(p) for p in z["pairs"]):
        for k in range(r.size):
            rows.append([pair, float(r[k]), float(z["energy"][i][k]), float(z["force_analytic"][i][k])])
        # invert noise_residual = MARGIN*sqrt(2)*noise/(2 dr)/max|F| back to the measured energy noise
        fa = z["force_analytic"][i]
        scale0 = max(np.abs(fa[np.isfinite(fa)]).max(), 1e-8) if np.isfinite(fa).any() else 1e-8
        nr = float(z["noise_residual"][i]) if "noise_residual" in z.files else 0.0
        noise[pair] = nr * 2.0 * float(r[1] - r[0]) * scale0 / (NOISE_MARGIN * math.sqrt(2.0))
    write_csv(out / "l1_pec.csv", ["pair", "r_A", "energy_eV", "force_eV_per_A"], rows,
              ["two-atom curves in vacuum; energy relative to the largest separation; force on atom 2 along +r"])
    s = np.load(data_dir / "l1_lattice_scan.npz")
    write_csv(out / "l1_lattice_scan.csv", ["scale", "volume_A3", "energy_eV_per_atom", "pressure_GPa", "min_distance_A"],
              [[float(s["scales"][k]), float(s["volumes"][k]), float(s["energy_per_atom"][k]),
                float(s["pressure_gpa"][k]), float(s["min_distance"][k])] for k in range(s["scales"].size)],
              ["unrelaxed isotropic linear scaling of the 306-atom probe cell"])
    for name, member in (("l1_nve_dt3", "dt=3 fs NVE, 1200 K start"), ("l1_nve_dt1.5", "dt=1.5 fs NVE, 1200 K start"),
                         ("l1_stability_2000K", "2000 K NVT, dt=3 fs")):
        m = np.load(data_dir / f"{name}.npz", allow_pickle=False)
        n_atoms = int(m["positions"].shape[1])
        etot = m["total_energy"] if "total_energy" in m.files else m["potential_energy"] + m["kinetic_energy"]
        crashed = int(m["crashed_at"])
        write_csv(out / f"{name}.csv",
                  ["time_fs", "etot_eV_per_atom", "temperature_K", "min_distance_A", "com_drift_A"],
                  [[float(m["time_fs"][k]), float(etot[k] / n_atoms), float(m["temperature"][k]),
                    float(m["min_distance"][k]), float(m["com_drift"][k])] for k in range(m["time_fs"].size)],
                  [member, f"completed={'false' if crashed >= 0 else 'true'}"])
    ps = ref.l1()["probe_structure"]
    probes = probe_record(l1_json["results"], n_atoms=ps["n_atoms"], volume_A3=ps["volume_A3"],
                          noise_eV=float(l1_json["context"]["measured_energy_noise_eV"]))
    probes["pec_energy_noise_eV"] = noise
    write_json(out / "l1_probes.json", {"format": "mlipeval-l1-probes/1", **probes})

    # L2
    relax = json.loads((data_dir / "l2_relax.json").read_text())
    key_of = {r["relax_label"]: r["key"] for r in structures}
    rows = []
    for label, rec in relax["relaxations"].items():
        if label not in key_of:
            continue  # extra cubic seeds relaxed only as EOS starting points
        a, b, c = rec["cell_lengths_A"]
        rows.append([key_of[label], rec["energy_per_atom_eV"], rec["volume_per_atom_A3"], a, b, c,
                     rec["max_force_eV_per_A"], rec["fmax"], int(rec["n_steps"]),
                     "true" if rec["converged"] else "false"])
    write_csv(out / "l2_relax.csv", ["key", "energy_eV_per_atom", "volume_A3_per_atom", "a_A", "b_A", "c_A",
                                     "max_force_eV_per_A", "fmax_eV_per_A", "n_steps", "converged"], rows,
              ["MLIP variable-cell relaxations of the reference structures (keys as in l2_reference.json)"])
    eos_rows = []
    for name in ["hex"] + [f"cubic_s{i:02d}" for i in range(1, 6)]:
        f = data_dir / f"l2_eos_{name}_mlip.npz"
        if not f.exists():
            continue
        e = np.load(f)
        n = int(e["n_atoms"])
        group, member = ("hex", "") if name == "hex" else ("cubic", name.split("_")[1])
        for v, en in zip(e["volumes"], e["energies"]):
            eos_rows.append([group, member, float(v) / n, float(en) / n])
    write_csv(out / "l2_eos.csv", ["group", "member", "volume_A3_per_atom", "energy_eV_per_atom"], eos_rows,
              ["MLIP fixed-volume, ion-relaxed EOS at the DFT volume scale factors"])

    # the HPC numbers the browser has to reproduce
    exp = {}
    for stem, payload in (("L0", audit), ("L1", l1_json), ("L2", json.loads(next(res_dir.glob("L2_*.json")).read_text()))):
        for rec in payload["results"]:
            v = rec["value"]
            scalar = isinstance(v, bool) or isinstance(v, (int, float)) or v is None
            # non-scalar values (PEC r_min/e_min, worst-case lists) are kept for their status and leakage flag
            exp[rec["metric_id"]] = [v if scalar else None, rec["status"], "training_leakage" in (rec.get("extra") or {})]
    exp["L1:gate_pass"] = [bool(l1_json["context"]["gate"]["pass"]), "context", False]
    expected[key] = exp
    overlap = sorted(m for m, v in exp.items() if v[2])
    if overlap:
        write_json(out / "training_overlap.json", {
            "format": "mlipeval-training-overlap/1",
            "note": "metrics whose reference structures are in this model's training set "
                    "(from the leakage screen); they are reported but not counted as passed tests",
            "metrics": overlap})

write_json(OUT / "expected_metrics.json", {"note": "HPC values of the scalar L0 (audit), L1 and L2 metrics; "
                                                   "[value, status, trained_on_this_reference]",
                                           "models": expected})
write_json(OUT / "index.json", {
    "reference": {"id": ref.reference_id, "files": {
        "l0_labels": "reference/l0_labels.extxyz.gz", "l1_reference": "reference/l1_reference.json",
        "l2_reference": "reference/l2_reference.json",
        "l2_eos": "reference/l2_eos.csv"}},
    "models": {k: {"files": {n: f"models/{k}/{n}" for n in (
        "l0_predictions.extxyz.gz", "l1_pec.csv", "l1_lattice_scan.csv", "l1_nve_dt3.csv", "l1_nve_dt1.5.csv",
        "l1_stability_2000K.csv", "l1_probes.json", "l2_relax.csv", "l2_eos.csv")},
        **({"training_overlap": f"models/{k}/training_overlap.json"} if (OUT / "models" / k / "training_overlap.json").exists() else {})}
        for k in SHOW},
})
total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
print(f"wrote {OUT} ({total / 2**20:.1f} MiB)")
