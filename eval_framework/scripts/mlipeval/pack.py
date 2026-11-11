"""Evaluation packs: move the model-side numbers off the cluster, score anywhere.

Two containers, both zip files with a ``manifest.json`` at the root that lists
every member with its sha256:

* **reference pack** (``*.mlipref``) -- the frozen DFT/AIMD side of L0-L2:
  the H5 audit labels and the static diagnostic labels (with their leakage
  screens), the AIMD NVE drift the L1 probe is compared with, and every DFT
  relaxation, static and EOS point L2 reads.  Built once from the phase-1 tree
  (:func:`build_reference`); after that nothing reads an OUTCAR.

* **model pack** (``*.mlippack``) -- what the HPC job computed with the model
  and nothing else: L0 predictions on the reference frames, the L1 dimer
  curves, lattice scan, MD probes and single-frame probe measurements, the L2
  relaxation records and EOS curves (:func:`build_model_pack`).  The original
  result files ride along under ``views/`` for display and for checking.

:mod:`mlipeval.rescore` turns a model pack plus a reference pack back into
L0/L1/L2 result files on a CPU, with the same code the drivers use, so a
threshold, a binning or a grading rule can change without going back to the
GPU.

The manifest binds a model pack to one reference pack (its id, the sha256 of
its manifest and of each label file) and to the frame fingerprints its
predictions were computed on.  :func:`verify_model_pack` refuses predictions
that do not line up with the reference frames.  What a pack cannot prove is
that the numbers came from the checkpoint it names -- only the sha256 it
declares is recorded -- so a platform should build packs in its own jobs rather
than accept hand-made ones.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import os
import platform
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PACK_SCHEMA = "mlipeval-pack/1"
REFERENCE_SCHEMA = "mlipeval-reference/1"
REFERENCE_ID = "gst225_phase1_v1"
MANIFEST = "manifest.json"

#: L2 relaxation labels -> key of the reference composition they relax.
_SPECIES = ("Ge", "Sb", "Te")
PEC_GRID = {"r_min": 1.0, "r_max": 8.0, "dr": 0.02}


class PackError(ValueError):
    """A pack is malformed, tampered with, or does not match its reference."""


# =========================================================================
# container helpers
# =========================================================================
def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            h.update(block)
    return h.hexdigest()


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _jsonable(obj: Any) -> Any:
    from .schema import _jsonable as base

    return base(obj)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _content_digest(files: Mapping[str, Mapping[str, Any]]) -> str:
    """Hash over (path, sha256) pairs: identifies the content, not the zip bytes."""
    h = hashlib.sha256()
    for name in sorted(files):
        h.update(name.encode())
        h.update(b"\0")
        h.update(str(files[name]["sha256"]).encode())
        h.update(b"\n")
    return h.hexdigest()


def write_container(staging: Path, out: Path, manifest: dict[str, Any]) -> Path:
    """Hash every file under ``staging``, write the manifest, zip it to ``out``."""
    staging = Path(staging)
    files = {}
    for p in sorted(staging.rglob("*")):
        if p.is_file() and p.name != MANIFEST:
            rel = p.relative_to(staging).as_posix()
            files[rel] = {"sha256": sha256_file(p), "size": p.stat().st_size}
    manifest = dict(manifest)
    manifest["files"] = files
    manifest["content_sha256"] = _content_digest(files)
    _write_json(staging / MANIFEST, manifest)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    with zipfile.ZipFile(tmp, "w") as zf:
        zf.write(staging / MANIFEST, MANIFEST, compress_type=zipfile.ZIP_DEFLATED)
        for rel in files:
            # npz members are already compressed; deflating them again buys nothing
            kind = zipfile.ZIP_STORED if rel.endswith(".npz") else zipfile.ZIP_DEFLATED
            zf.write(staging / rel, rel, compress_type=kind)
    tmp.replace(out)
    return out


def _safe_member(name: str) -> bool:
    p = Path(name)
    return not (p.is_absolute() or ".." in p.parts or name.startswith(("/", "\\")))


MAX_MEMBERS = 10_000


def max_unpacked_bytes() -> int:
    return int(float(os.environ.get("MLIPEVAL_MAX_UNPACKED_GB", "8")) * 2**30)


def unpack_root() -> Path:
    root = os.environ.get("MLIPEVAL_PACK_CACHE")
    return Path(root).expanduser() if root else Path.home() / ".cache" / "mlipeval" / "unpacked"


def read_manifest(source: str | os.PathLike) -> dict[str, Any]:
    """Manifest of a container (zip) or an unpacked directory, without extracting."""
    source = Path(source)
    if source.is_dir():
        return json.loads((source / MANIFEST).read_text(encoding="utf-8"))
    with zipfile.ZipFile(source) as zf:
        try:
            return json.loads(zf.read(MANIFEST).decode("utf-8"))
        except KeyError:
            raise PackError(f"{source}: no {MANIFEST} in the archive") from None


def verify_files(root: Path, manifest: Mapping[str, Any]) -> list[str]:
    """Every listed file present with the listed sha256, and nothing unlisted."""
    errors = []
    listed = manifest.get("files", {})
    for rel, meta in listed.items():
        p = root / rel
        if not p.is_file():
            errors.append(f"missing member {rel}")
        elif sha256_file(p) != meta["sha256"]:
            errors.append(f"sha256 mismatch for {rel}")
    for p in root.rglob("*"):
        if p.is_file() and p.name != MANIFEST and p.relative_to(root).as_posix() not in listed:
            errors.append(f"unlisted member {p.relative_to(root).as_posix()}")
    if manifest.get("content_sha256") and manifest["content_sha256"] != _content_digest(listed):
        errors.append("content_sha256 does not match the file list")
    return errors


def extract(source: str | os.PathLike, dest: str | os.PathLike | None = None) -> Path:
    """Unpack a container (idempotent, verified) and return the directory.

    Without ``dest`` the pack goes to ``$MLIPEVAL_PACK_CACHE`` (default
    ``~/.cache/mlipeval/unpacked``) under its content hash, so opening the same
    pack twice costs one hash check.  A directory is verified in place.
    """
    source = Path(source)
    if source.is_dir():
        manifest = read_manifest(source)
        errors = verify_files(source, manifest)
        if errors:
            raise PackError(f"{source}: " + "; ".join(errors[:5]))
        return source

    with zipfile.ZipFile(source) as zf:
        try:
            raw = zf.read(MANIFEST)
        except KeyError:
            raise PackError(f"{source}: no {MANIFEST} in the archive") from None
    manifest = json.loads(raw.decode("utf-8"))
    # keyed by the manifest bytes, not only the file list: two packs with the
    # same members but different bindings (reference, model spec) must not share
    # an unpacked directory
    digest = hashlib.sha256(raw).hexdigest()
    target = Path(dest) if dest else unpack_root() / f"{manifest.get('kind', 'pack')}_{digest[:16]}"
    if (target / MANIFEST).exists() and not verify_files(target, read_manifest(target)):
        return target
    if target.exists():
        shutil.rmtree(target)
    # unpack beside the target so the final rename never crosses filesystems
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".unpack_", dir=str(target.parent)))
    try:
        with zipfile.ZipFile(source) as zf:
            infos = zf.infolist()
            if len(infos) > MAX_MEMBERS:
                raise PackError(f"{source}: {len(infos)} members, more than {MAX_MEMBERS}")
            total = sum(i.file_size for i in infos)
            if total > max_unpacked_bytes():
                raise PackError(f"{source}: unpacks to {total / 2**30:.1f} GiB, more than the "
                                f"{max_unpacked_bytes() / 2**30:.1f} GiB limit (MLIPEVAL_MAX_UNPACKED_GB)")
            for info in infos:
                if not _safe_member(info.filename):
                    raise PackError(f"{source}: unsafe member path {info.filename!r}")
            zf.extractall(tmp)
        errors = verify_files(tmp, read_manifest(tmp))
        if errors:
            raise PackError(f"{source}: " + "; ".join(errors[:5]))
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp.replace(target)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    return target


# =========================================================================
# reference pack
# =========================================================================
#: DFT tasks L2 reads, by role.  Missing ones are recorded, not fatal.
def _l2_tasks() -> list[str]:
    from .datasets import ELEMENTS, PHASE_LADDER

    tasks = []
    for _key, _src, static, _label in PHASE_LADDER:
        if "{seed" in static:
            continue
        tasks += [static.replace("_static", "_relax"), static]
    for seed in range(1, 6):
        tasks += [f"H1_cubic_s{seed:02d}_relax", f"H1_cubic_s{seed:02d}_static"]
    for el in ELEMENTS:
        tasks += [f"L2_{el.lower()}_relax", f"L2_{el.lower()}_static"]
    return tasks


def _composition(atoms) -> dict[str, int]:
    symbols = atoms.get_chemical_symbols()
    return {s: symbols.count(s) for s in sorted(set(symbols))}


def build_reference(
    out: str | os.PathLike,
    *,
    include_diagnostic: bool = True,
    fallback_model: str | None = None,
) -> Path:
    """Freeze the phase-1 reference side of L0-L2 into one container.

    Reads the harvested label files under ``data/reference`` and the phase-1
    VASP tree (``paths.PHASE1``); run it where both are -- on hopper.

    ``fallback_model`` is for a partial copy of the phase-1 tree (the local
    mirror holds only the files the notebooks open).  DFT numbers whose OUTCAR is
    absent are then taken from what ``run_l2.py`` already harvested from the same
    OUTCARs for that model: the ``l2_eos_*_dft.npz`` curves and the element
    chemical potentials in ``L2_<key>.json``.  Every such value is listed under
    ``provenance.fallbacks`` and the manifest is marked ``complete: false``; a
    pack meant for distribution should be built without it.
    """
    from . import datasets, paths, protocols
    from .cache import frames_fingerprint
    from .io_vasp import read_poscar
    from .l2_static import load_dft_eos

    out = Path(out)
    with tempfile.TemporaryDirectory(prefix="mlipref_") as tmp:
        stage = Path(tmp)
        levels: dict[str, Any] = {}
        missing: list[str] = []
        fallbacks: list[str] = []
        fb_data = paths.MODEL_DATA_ROOT / fallback_model if fallback_model else None
        fb_l2 = None
        if fallback_model:
            hits = sorted((paths.MODEL_RESULTS_ROOT / fallback_model).glob("L2_*.json"))
            fb_l2 = json.loads(hits[0].read_text()) if hits else None

        # ---------------------------------------------------------------- L0
        (stage / "L0").mkdir()
        shutil.copy2(paths.H5_LABELS, stage / "L0/h5_audit.extxyz")
        shutil.copy2(paths.H5_LEAKAGE, stage / "L0/h5_audit.leakage.json")
        h5_manifest = paths.H5_LABELS.with_suffix(".manifest.json")
        if h5_manifest.exists():
            shutil.copy2(h5_manifest, stage / "L0/h5_audit.harvest.json")
        h5 = datasets.labelled_h5()
        levels["L0"] = {"audit": {
            "file": "L0/h5_audit.extxyz",
            "sha256": sha256_file(paths.H5_LABELS),
            "n_frames": len(h5),
            "fingerprint": frames_fingerprint(h5.frames),
            "role": h5.role.value,
            "protocol_id": h5.protocol_id,
        }}
        if include_diagnostic:
            shutil.copy2(paths.DIAGNOSTIC_LABELS, stage / "L0/diagnostic.extxyz")
            shutil.copy2(paths.DIAGNOSTIC_LEAKAGE, stage / "L0/diagnostic.leakage.json")
            diag_manifest = paths.DIAGNOSTIC_LABELS.with_suffix(".manifest.json")
            if diag_manifest.exists():
                shutil.copy2(diag_manifest, stage / "L0/diagnostic.harvest.json")
            diag = datasets.labelled_diagnostic()
            levels["L0"]["diagnostic"] = {
                "file": "L0/diagnostic.extxyz",
                "sha256": sha256_file(paths.DIAGNOSTIC_LABELS),
                "n_frames": len(diag),
                "fingerprint": frames_fingerprint(diag.frames),
                "role": diag.role.value,
                "protocol_id": diag.protocol_id,
            }

        # ---------------------------------------------------------------- L1
        probe_dir = paths.PHASE1 / datasets.PREFLIGHT["nve_1200K"]
        start = read_poscar(probe_dir / "POSCAR")
        (stage / "L1").mkdir()
        shutil.copy2(probe_dir / "POSCAR", stage / "L1/probe_POSCAR")
        l1 = {
            "aimd_nve_reference": datasets.preflight_nve_drift(),
            "probe_structure": {
                "task": datasets.PREFLIGHT["nve_1200K"],
                "file": "L1/probe_POSCAR",
                "n_atoms": len(start),
                "volume_A3": float(start.get_volume()),
                "composition": _composition(start),
            },
            "probe_protocol": protocols.canonical_aimd().to_dict(),
        }
        _write_json(stage / "L1/reference.json", l1)
        levels["L1"] = {"file": "L1/reference.json"}

        # ---------------------------------------------------------------- L2
        tasks = {}
        for task in _l2_tasks():
            try:
                rec = datasets.dft_relaxed(task)
            except FileNotFoundError as exc:
                mu = _fallback_mu(task, fb_l2)
                if mu is not None:
                    tasks[task] = {"task": task, "protocol_id": "gst225_pbe_canonical_v1",
                                   "energy_per_atom_eV": mu, "partial": True,
                                   "source": f"mu_dft in L2_{fallback_model}.json (OUTCAR not in this tree)"}
                    fallbacks.append(f"{task}: energy_per_atom_eV from L2_{fallback_model}.json")
                else:
                    missing.append(f"{task}: {exc}")
                continue
            atoms = rec.pop("atoms")
            rec["composition"] = _composition(atoms)
            tasks[task] = rec
        compositions = {}
        for key, source, _static, _label in datasets.PHASE_LADDER:
            if source != "cubic_seed":
                compositions[key] = _composition(datasets.mp_structure(source))
        for seed in range(1, 6):
            compositions[f"cubic_s{seed:02d}"] = _composition(datasets.cubic_seed(seed))
        for el in datasets.ELEMENTS:
            compositions[el] = _composition(datasets.element_reference(el))
        eos = {}
        for phase in ["hex"] + [f"cub{s:02d}" for s in range(1, 6)]:
            task_dirs = datasets.dft_eos_tasks(phase)
            name = "hex" if phase == "hex" else f"cubic_s{phase[3:]}"
            curve = load_dft_eos(task_dirs, label=f"{phase}_dft") if task_dirs else None
            fb = fb_data / f"l2_eos_{name}_dft.npz" if fb_data else None
            if fb is not None and fb.exists():
                z = np.load(fb, allow_pickle=False)
                # a partial tree yields fewer points than run_l2 harvested on the cluster
                if curve is None or len(z["volumes"]) > len(curve.volumes):
                    eos[phase] = {"volumes_A3": z["volumes"].tolist(), "energies_eV": z["energies"].tolist(),
                                  "n_atoms": int(z["n_atoms"]), "source": str(z["source"]), "skipped": [],
                                  "scale_kind": "volume", "fallback": str(fb)}
                    have = 0 if curve is None else len(curve.volumes)
                    fallbacks.append(f"EOS {phase}: {len(z['volumes'])}-point curve from {fb.name} "
                                     f"({fallback_model}); {have} point(s) in this tree")
                    continue
            if curve is None:
                missing.append(f"EOS {phase}: no completed tasks")
                continue
            eos[phase] = {
                "volumes_A3": curve.volumes.tolist(),
                "energies_eV": curve.energies.tolist(),
                "n_atoms": int(curve.n_atoms),
                "source": curve.source,
                "skipped": list(getattr(curve, "skipped", [])),
                "scale_kind": "volume",
            }
        l2 = {
            "dft_protocol": "gst225_pbe_canonical_v1",
            "tasks": tasks,
            "compositions": compositions,
            "phase_ladder": [list(row) for row in datasets.PHASE_LADDER],
            "elements": list(datasets.ELEMENTS),
            "eos": eos,
        }
        (stage / "L2").mkdir()
        _write_json(stage / "L2/reference.json", l2)
        levels["L2"] = {"file": "L2/reference.json", "n_tasks": len(tasks), "eos": sorted(eos)}

        manifest = {
            "schema": REFERENCE_SCHEMA,
            "kind": "reference",
            "reference_id": REFERENCE_ID,
            "system": "GST225 (Ge2Sb2Te5)",
            "protocols": {
                "static": protocols.canonical_static().to_dict(),
                "aimd": protocols.canonical_aimd().to_dict(),
            },
            "levels": levels,
            "complete": not missing and not fallbacks,
            "missing": missing,
            "provenance": {
                "fallbacks": fallbacks,
                "built_at": _now(),
                "host": platform.node(),
                "phase1": str(paths.PHASE1),
                "reference_dir": str(paths.REFERENCE),
            },
        }
        return write_container(stage, out, manifest)


def _fallback_mu(task: str, l2_json: Mapping[str, Any] | None) -> float | None:
    """Element chemical potential of ``L2_<el>_static`` from an L2 result file."""
    m = re.fullmatch(r"L2_([a-z]+)_static", task)
    if not m or l2_json is None:
        return None
    element = m.group(1).capitalize()
    for r in l2_json["results"]:
        mu = (r.get("extra") or {}).get("mu_dft_eV_per_atom")
        if mu and element in mu:
            return float(mu[element])
    return None


@dataclass
class Reference:
    """An unpacked, verified reference pack."""

    root: Path
    manifest: dict[str, Any]

    @property
    def reference_id(self) -> str:
        return str(self.manifest["reference_id"])

    @property
    def manifest_sha256(self) -> str:
        return sha256_file(self.root / MANIFEST)

    def path(self, rel: str) -> Path:
        return self.root / rel

    def has_diagnostic(self) -> bool:
        return "diagnostic" in self.manifest["levels"].get("L0", {})

    def h5(self):
        from .datasets import labelled_h5

        return labelled_h5(path=self.path("L0/h5_audit.extxyz"),
                           leakage=self.path("L0/h5_audit.leakage.json"))

    def h5_leakage(self) -> dict[str, Any]:
        return json.loads(self.path("L0/h5_audit.leakage.json").read_text())

    def diagnostic(self):
        if not self.has_diagnostic():
            raise PackError("this reference pack was built without the diagnostic set")
        from .datasets import labelled_diagnostic

        return labelled_diagnostic(self.path("L0/diagnostic.extxyz"))

    def diagnostic_overlap(self, labelled, model_key: str) -> np.ndarray:
        from .datasets import diagnostic_training_overlap

        return diagnostic_training_overlap(
            labelled, model_key,
            labels=self.path("L0/diagnostic.extxyz"),
            leakage=self.path("L0/diagnostic.leakage.json"),
        )

    def l1(self) -> dict[str, Any]:
        return json.loads(self.path("L1/reference.json").read_text())

    def l2(self) -> dict[str, Any]:
        return json.loads(self.path("L2/reference.json").read_text())

    def eos_curve(self, phase: str):
        from .l2_static import EOSCurve

        rec = self.l2()["eos"][phase]
        curve = EOSCurve(label=f"{phase}_dft", volumes=np.asarray(rec["volumes_A3"], dtype=float),
                         energies=np.asarray(rec["energies_eV"], dtype=float),
                         n_atoms=int(rec["n_atoms"]), source=rec["source"])
        curve.skipped = rec.get("skipped", [])  # type: ignore[attr-defined]
        return curve


def open_reference(source: str | os.PathLike) -> Reference:
    root = extract(source)
    manifest = read_manifest(root)
    if manifest.get("schema") != REFERENCE_SCHEMA:
        raise PackError(f"{source}: schema {manifest.get('schema')!r}, expected {REFERENCE_SCHEMA}")
    return Reference(root=root, manifest=manifest)


def default_reference_path() -> Path:
    """``$MLIPEVAL_REFERENCE``, else ``data/reference/<id>.mlipref`` under the output root."""
    env = os.environ.get("MLIPEVAL_REFERENCE")
    if env:
        return Path(env).expanduser()
    from . import paths

    return paths.REFERENCE / f"{REFERENCE_ID}.mlipref"


# =========================================================================
# model pack
# =========================================================================
def _load_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def _first_result(results_dir: Path, pattern: str) -> tuple[Path, dict[str, Any]] | None:
    hits = sorted(results_dir.glob(pattern))
    return (hits[0], json.loads(hits[0].read_text())) if hits else None


def spec_from_dict(d: Mapping[str, Any]):
    from .schema import ModelSpec

    allowed = {f.name for f in fields(ModelSpec)}
    return ModelSpec(**{k: v for k, v in d.items() if k in allowed})


def _find_cache(cache_dir: Path, tag: str, spec: Mapping[str, Any], fingerprint: str) -> Path | None:
    """The prediction cache entry for (tag, checkpoint, artefact, frames), any device."""
    sha10 = spec["checkpoint_sha256"][:10]
    artifact = (spec.get("software") or {}).get("artifact", "")
    hits = sorted(cache_dir.glob(f"{tag}__{sha10}__*{fingerprint}.npz"))
    if artifact:
        hits = [h for h in hits if f"__{artifact}__" in h.name]
    if not hits:
        return None
    preferred = f"__{spec.get('device')}_{spec.get('dtype')}"
    hits.sort(key=lambda h: (preferred not in h.name, h.name))
    return hits[0]


def _find_pec(cache_dir: Path, spec: Mapping[str, Any]) -> Path | None:
    sha10 = spec["checkpoint_sha256"][:10]
    artifact = (spec.get("software") or {}).get("artifact", "")
    g = PEC_GRID
    key = f"pec2__{sha10}__{artifact}__{'-'.join(_SPECIES)}__{g['r_min']:g}_{g['r_max']:g}_{g['dr']:g}"
    hits = sorted(cache_dir.glob(f"{key}__*.npz"))
    preferred = f"__{spec.get('device')}_{spec.get('dtype')}.npz"
    hits.sort(key=lambda h: (not h.name.endswith(preferred), h.name))
    return hits[0] if hits else None


def _relax_label_key(label: str) -> str:
    """``hex_mp1224375`` -> ``mp1224375``, ``phase_mp3534`` -> ``mp3534``,
    ``element_Ge`` -> ``Ge``, ``cubic_s01`` -> ``cubic_s01``."""
    if label.startswith("hex_"):
        return label[4:]
    if label.startswith(("phase_", "element_")):
        return label.split("_", 1)[1]
    return label


def _backfill_relax(l2_json: Mapping[str, Any], ref_l2: Mapping[str, Any]) -> dict[str, Any]:
    """Relaxation records rebuilt from an ``L2_<key>.json`` written before
    ``run_l2.py`` saved ``l2_relax.json``.  Cell lengths/angles come back rounded
    to 1e-4 A / 1e-3 deg (that is all the result file kept)."""
    fmax = float(l2_json["context"].get("fmax_eV_per_A", 0.01))
    records, cubic_seed = {}, None
    for r in l2_json["results"]:
        m = re.fullmatch(r"L2\.relax\.(.+)\.energy_per_atom", r["metric_id"])
        if not m:
            continue
        label = m.group(1)
        summ = r["extra"]
        comp_key = _relax_label_key(label)
        composition = ref_l2["compositions"].get(comp_key)
        if composition is None:
            raise PackError(f"no reference composition for relaxation {label!r}")
        n = int(summ["n_atoms"])
        records[label] = {
            "label": label,
            "n_atoms": n,
            "composition": composition,
            "energy_eV": float(summ["energy_per_atom_eV"]) * n,
            "energy_per_atom_eV": float(summ["energy_per_atom_eV"]),
            "volume_A3": float(summ["volume_A3"]),
            "volume_per_atom_A3": float(summ["volume_per_atom_A3"]),
            "volume_change_pct": float(summ["volume_change_pct"]),
            "cell_lengths_A": list(summ["cell_lengths_A"]),
            "cell_angles_deg": list(summ["cell_angles_deg"]),
            "max_force_eV_per_A": float(summ["max_force_eV_per_A"]),
            "fmax": fmax,
            "n_steps": int(summ["n_steps"]),
            "converged": bool(summ["converged"]),
        }
        if label.startswith("cubic_s"):
            cubic_seed = int(label[len("cubic_s"):])
    ctx = l2_json["context"]
    return {
        "fmax_eV_per_A": fmax,
        "cubic_seed": cubic_seed or 1,
        "eos_scales": ctx.get("eos_scales", [0.92, 0.95, 0.98, 1.0, 1.02, 1.05, 1.08]),
        # files without the key predate the 2026-11-11 fix: their scan was linear
        "eos_scale_kind": ctx.get("eos_scale_kind", "linear"),
        "relaxations": records,
        "backfilled_from": "L2 result file (cell lengths rounded to 1e-4 A)",
    }


def build_model_pack(
    model_key: str,
    out: str | os.PathLike,
    *,
    reference: Reference | str | os.PathLike | None = None,
    results_dir: str | os.PathLike | None = None,
    data_dir: str | os.PathLike | None = None,
    cache_dir: str | os.PathLike | None = None,
    include_views: bool = True,
) -> tuple[Path, dict[str, Any]]:
    """Collect one model's L0-L2 raw outputs into a pack.

    Works on what the drivers already left on disk -- prediction caches, the
    per-model ``data/models/<key>`` npz files and the result JSONs -- so packs
    can be built for every model that has been run, without new GPU time.
    Files that newer drivers write (``l1_probes.json``, ``l2_relax.json``) are
    used when present; otherwise the same numbers are taken from the result
    files and the manifest says so under ``provenance.backfilled``.

    Returns the container path and a summary of what went in.
    """
    from . import paths
    from .cache import frames_fingerprint, load_predictions
    from .l1_pes import probe_record

    ref = reference if isinstance(reference, Reference) else open_reference(reference or default_reference_path())
    results_dir = Path(results_dir) if results_dir else paths.MODEL_RESULTS_ROOT / model_key
    data_dir = Path(data_dir) if data_dir else paths.MODEL_DATA_ROOT / model_key
    cache_dir = Path(cache_dir) if cache_dir else paths.CACHE
    if not results_dir.is_dir():
        raise PackError(f"no results for {model_key!r} under {results_dir}")

    jsons = {
        "L0_audit": _load_json(results_dir / "L0_audit.json"),
        "L0_diagnostic": _load_json(results_dir / "L0_diagnostic.json"),
        "L1": (_first_result(results_dir, "L1_*.json") or (None, None))[1],
        "L2": (_first_result(results_dir, "L2_*.json") or (None, None))[1],
    }
    specs = {k: v["model"] for k, v in jsons.items() if v and v.get("model")}
    if not specs:
        raise PackError(f"{model_key}: no L0/L1/L2 result file carries a model spec")
    shas = {k: s["checkpoint_sha256"] for k, s in specs.items()}
    if len(set(shas.values())) > 1:
        raise PackError(f"{model_key}: result files name different checkpoints: {shas}")
    spec = next(iter(specs.values()))

    summary: dict[str, Any] = {"model_key": model_key, "levels": {}, "backfilled": [], "absent": []}
    levels: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="mlippack_") as tmp:
        stage = Path(tmp)

        # ---------------------------------------------------------------- L0
        l0: dict[str, Any] = {}
        h5 = ref.h5()
        fp = frames_fingerprint(h5.frames)
        hit = _find_cache(cache_dir, f"L0_h5_{len(h5)}", spec, fp)
        if hit is not None:
            pred = load_predictions(hit)
            if len(pred.energies) != len(h5):
                raise PackError(f"{hit.name}: {len(pred.energies)} predictions for {len(h5)} frames")
            (stage / "L0").mkdir(exist_ok=True)
            shutil.copy2(hit, stage / "L0/audit_predictions.npz")
            l0["audit"] = {"file": "L0/audit_predictions.npz", "n_frames": len(h5), "fingerprint": fp,
                           "source": hit.name}
        else:
            summary["absent"].append("L0 audit predictions (no cache entry)")

        if ref.has_diagnostic():
            diag = ref.diagnostic()
            seen = ref.diagnostic_overlap(diag, model_key)
            kept = np.flatnonzero(~seen)
            subset = diag.subset(~seen)
            fp_d = frames_fingerprint(subset.frames)
            hit = _find_cache(cache_dir, f"L0_diagnostic_{len(subset)}", spec, fp_d)
            if hit is not None:
                pred = load_predictions(hit)
                if len(pred.energies) != len(subset):
                    raise PackError(f"{hit.name}: {len(pred.energies)} predictions for {len(subset)} frames")
                (stage / "L0").mkdir(exist_ok=True)
                shutil.copy2(hit, stage / "L0/diagnostic_predictions.npz")
                _write_json(stage / "L0/diagnostic_frames.json", {
                    "n_reference_frames": len(diag),
                    "kept_indices": kept.tolist(),
                    "training_overlap_removed": int(seen.sum()),
                    "fingerprint": fp_d,
                })
                l0["diagnostic"] = {"file": "L0/diagnostic_predictions.npz", "frames": "L0/diagnostic_frames.json",
                                    "n_frames": len(subset), "fingerprint": fp_d, "source": hit.name}
            else:
                summary["absent"].append("L0 diagnostic predictions (no cache entry)")
        if l0:
            levels["L0"] = l0

        # ---------------------------------------------------------------- L1
        l1: dict[str, Any] = {}
        l1_json = jsons["L1"]
        (stage / "L1").mkdir()
        pec = _find_pec(cache_dir, spec)
        if pec is not None:
            shutil.copy2(pec, stage / "L1/pec.npz")
            l1["pec"] = {"file": "L1/pec.npz", "source": pec.name, "grid": PEC_GRID}
        else:
            summary["absent"].append("L1 PEC (no cache entry)")
        for name, member in (("l1_lattice_scan.npz", "lattice_scan"), ("l1_nve_dt3.npz", "nve_dt3"),
                             ("l1_nve_dt1.5.npz", "nve_dt1.5"), ("l1_stability_2000K.npz", "stability_2000K")):
            if (data_dir / name).exists():
                shutil.copy2(data_dir / name, stage / f"L1/{member}.npz")
                l1[member] = {"file": f"L1/{member}.npz"}
            else:
                summary["absent"].append(f"L1 {name}")
        probes_file = data_dir / "l1_probes.json"
        probe_struct = ref.l1()["probe_structure"]
        if probes_file.exists():
            probes = json.loads(probes_file.read_text())
            probes_source = "l1_probes.json"
        elif l1_json is not None:
            probes = probe_record(l1_json["results"], n_atoms=probe_struct["n_atoms"],
                                  volume_A3=probe_struct["volume_A3"],
                                  noise_eV=float(l1_json["context"]["measured_energy_noise_eV"]))
            probes_source = "backfilled from the L1 result file"
            summary["backfilled"].append("L1 probes")
        else:
            probes, probes_source = None, None
            summary["absent"].append("L1 probes")
        if probes is not None:
            _write_json(stage / "L1/probes.json", probes)
            l1["probes"] = {"file": "L1/probes.json", "source": probes_source}
        if l1_json is not None:
            ctx = l1_json["context"]
            settings = dict(ctx.get("md_probes") or {})
            if not settings:
                spread = next((r for r in l1_json["results"]
                               if r["metric_id"] == "L1.timestep_sensitivity.drift_spread"), None)
                settings = {
                    "stability_temperature_K": 2000.0,
                    "timestep_total_time_fs": (spread or {}).get("extra", {}).get(
                        "total_time_fs", 300.0 if ctx.get("quick") else 999.0),
                    "timesteps_fs": [3.0, 1.5],
                }
            settings["quick"] = bool(ctx.get("quick", False))
            settings["skip_md"] = not any(r["metric_id"].startswith("L1.nve_drift") for r in l1_json["results"])
            l1["settings"] = settings
        if l1:
            levels["L1"] = l1

        # ---------------------------------------------------------------- L2
        l2: dict[str, Any] = {}
        relax_file = data_dir / "l2_relax.json"
        if relax_file.exists():
            relax = json.loads(relax_file.read_text())
            relax_source = "l2_relax.json"
        elif jsons["L2"] is not None:
            relax = _backfill_relax(jsons["L2"], ref.l2())
            relax_source = "backfilled from the L2 result file"
            summary["backfilled"].append("L2 relaxations")
        else:
            relax, relax_source = None, None
            summary["absent"].append("L2 relaxations")
        if relax is not None:
            (stage / "L2").mkdir()
            _write_json(stage / "L2/relax.json", relax)
            l2["relax"] = {"file": "L2/relax.json", "source": relax_source,
                           "labels": sorted(relax["relaxations"])}
            eos = {}
            for name in ["hex"] + [f"cubic_s{s:02d}" for s in range(1, 6)]:
                src = data_dir / f"l2_eos_{name}_mlip.npz"
                if src.exists():
                    shutil.copy2(src, stage / f"L2/eos_{name}.npz")
                    eos[name] = f"L2/eos_{name}.npz"
            l2["eos"] = eos
            l2["eos_scale_kind"] = relax.get("eos_scale_kind", "linear")
            levels["L2"] = l2

        # ------------------------------------------------------------- views
        views = []
        if include_views:
            (stage / "views").mkdir()
            for p in sorted(results_dir.glob("*.json")):
                shutil.copy2(p, stage / "views" / p.name)
                views.append(f"views/{p.name}")
            for pattern in ("l3_*.npz", "l4_*.npz", "l5_*.npz"):
                for p in sorted(data_dir.glob(pattern)):
                    shutil.copy2(p, stage / "views" / p.name)
                    views.append(f"views/{p.name}")

        try:
            from . import models

            label = models.get(model_key).label
        except Exception:
            label = spec.get("name", model_key)
        manifest = {
            "schema": PACK_SCHEMA,
            "kind": "model",
            "model_key": model_key,
            "label": label,
            "model": spec,
            "reference": {
                "reference_id": ref.reference_id,
                "manifest_sha256": ref.manifest_sha256,
                "content_sha256": ref.manifest.get("content_sha256"),
                "labels_sha256": {k: v["sha256"] for k, v in ref.manifest["levels"]["L0"].items()},
            },
            "levels": levels,
            "views": views,
            "provenance": {
                "built_at": _now(),
                "host": platform.node(),
                "results_dir": str(results_dir),
                "data_dir": str(data_dir),
                "cache_dir": str(cache_dir),
                "backfilled": summary["backfilled"],
                "absent": summary["absent"],
            },
        }
        path = write_container(stage, Path(out), manifest)
    summary["levels"] = {k: sorted(v) for k, v in levels.items()}
    summary["path"] = str(path)
    return path, summary


@dataclass
class ModelPack:
    """An unpacked, hash-verified model pack."""

    root: Path
    manifest: dict[str, Any]

    @property
    def model_key(self) -> str:
        return str(self.manifest["model_key"])

    @property
    def pack_id(self) -> str:
        return str(self.manifest["content_sha256"])[:16]

    @property
    def spec(self):
        return spec_from_dict(self.manifest["model"])

    def level(self, name: str) -> dict[str, Any]:
        return self.manifest.get("levels", {}).get(name, {})

    def path(self, rel: str) -> Path:
        return self.root / rel

    def json(self, rel: str) -> Any:
        return json.loads(self.path(rel).read_text())

    def npz(self, rel: str):
        return np.load(self.path(rel), allow_pickle=False)

    def view(self, name: str) -> dict[str, Any] | None:
        p = self.path(f"views/{name}")
        return json.loads(p.read_text()) if p.exists() else None

    def original(self, result_name: str) -> dict[str, Any] | None:
        """The HPC result file a rescored ``result_name`` should reproduce.

        ``L0_audit.json`` / ``L0_diagnostic.json`` by name; ``L1``/``L2`` by
        prefix, since those files are named after the model key.
        """
        if result_name in ("L1", "L2"):
            hits = sorted(self.path("views").glob(f"{result_name}_*.json")) if self.path("views").is_dir() else []
            return json.loads(hits[0].read_text()) if hits else None
        return self.view(result_name)


def open_model_pack(source: str | os.PathLike) -> ModelPack:
    root = extract(source)
    manifest = read_manifest(root)
    if manifest.get("schema") != PACK_SCHEMA:
        raise PackError(f"{source}: schema {manifest.get('schema')!r}, expected {PACK_SCHEMA}")
    return ModelPack(root=root, manifest=manifest)


# =========================================================================
# verification
# =========================================================================
def verify_model_pack(pack: ModelPack, ref: Reference) -> dict[str, Any]:
    """Check that a pack can be scored against ``ref``; nothing is computed.

    ``errors`` block rescoring of the affected level; ``warnings`` are attached to
    its results.  File hashes were already checked when the pack was opened.
    """
    from .cache import frames_fingerprint, load_predictions

    checks: list[dict[str, Any]] = []

    def check(cid: str, ok: bool, detail: str, *, level: str = "-", severity: str = "error") -> None:
        checks.append({"id": cid, "level": level, "ok": bool(ok),
                       "severity": "info" if ok else severity, "detail": detail})

    want = pack.manifest.get("reference", {})
    check("reference.id", want.get("reference_id") == ref.reference_id,
          f"pack built against {want.get('reference_id')!r}, reference is {ref.reference_id!r}")
    check("reference.content", want.get("content_sha256") == ref.manifest.get("content_sha256"),
          "reference content hash matches" if want.get("content_sha256") == ref.manifest.get("content_sha256")
          else "the reference pack differs from the one this pack was built against",
          severity="warning")
    for name, sha in (want.get("labels_sha256") or {}).items():
        have = ref.manifest["levels"]["L0"].get(name, {}).get("sha256")
        check(f"reference.labels.{name}", have == sha,
              f"{name} labels sha256 {'match' if have == sha else 'differ'}", level="L0")

    check("reference.complete", ref.manifest.get("complete", True),
          "reference built from the full phase-1 tree" if ref.manifest.get("complete", True) else
          "reference has fallback values or missing tasks: "
          + "; ".join((ref.manifest.get("provenance", {}).get("fallbacks") or []) + ref.manifest.get("missing", [])),
          severity="warning")
    spec = pack.manifest.get("model") or {}
    check("model.checkpoint_sha256", bool(re.fullmatch(r"[0-9a-f]{64}", str(spec.get("checkpoint_sha256", "")))),
          f"declared checkpoint sha256 {str(spec.get('checkpoint_sha256', ''))[:16]}…; provenance is "
          "declared, not proven", severity="error")

    l0 = pack.level("L0")
    if "audit" in l0:
        h5 = ref.h5()
        pred = load_predictions(pack.path(l0["audit"]["file"]))
        fp = frames_fingerprint(h5.frames)
        check("L0.audit.fingerprint", l0["audit"]["fingerprint"] == fp,
              "predictions were computed on these H5 frames" if l0["audit"]["fingerprint"] == fp
              else "predictions belong to a different frame set", level="L0")
        same_atoms = len(pred.n_atoms) == len(h5) and np.array_equal(np.asarray(pred.n_atoms), h5.n_atoms)
        check("L0.audit.shape", same_atoms, f"{len(pred.energies)} predictions, {len(h5)} frames", level="L0")
    if "diagnostic" in l0:
        if not ref.has_diagnostic():
            check("L0.diagnostic.reference", False, "reference pack has no diagnostic set", level="L0")
        else:
            diag = ref.diagnostic()
            frames = pack.json(l0["diagnostic"]["frames"])
            kept = np.asarray(frames["kept_indices"], dtype=int)
            ok_range = frames["n_reference_frames"] == len(diag) and (kept.size == 0 or kept.max() < len(diag))
            check("L0.diagnostic.indices", ok_range,
                  f"{kept.size} of {len(diag)} frames kept", level="L0")
            if ok_range:
                mask = np.zeros(len(diag), dtype=bool)
                mask[kept] = True
                subset = diag.subset(mask)
                fp = frames_fingerprint(subset.frames)
                check("L0.diagnostic.fingerprint", fp == frames["fingerprint"],
                      "predictions were computed on these frames" if fp == frames["fingerprint"]
                      else "predictions belong to a different frame set", level="L0")
                expected = ref.diagnostic_overlap(diag, pack.model_key)
                check("L0.diagnostic.training_overlap", np.array_equal(~expected, mask),
                      "frames removed for training overlap agree with the reference screen"
                      if np.array_equal(~expected, mask)
                      else "the removed frames differ from the reference leakage screen", level="L0",
                      severity="warning")
                pred = load_predictions(pack.path(l0["diagnostic"]["file"]))
                same = len(pred.n_atoms) == len(subset) and np.array_equal(np.asarray(pred.n_atoms), subset.n_atoms)
                check("L0.diagnostic.shape", same, f"{len(pred.energies)} predictions, {len(subset)} frames",
                      level="L0")

    l1 = pack.level("L1")
    if l1:
        for member in ("pec", "probes", "lattice_scan"):
            check(f"L1.{member}", member in l1, f"{member} {'present' if member in l1 else 'missing'}",
                  level="L1", severity="warning")
        if not l1.get("settings", {}).get("skip_md", False):
            for member in ("nve_dt3", "nve_dt1.5", "stability_2000K"):
                check(f"L1.{member}", member in l1, f"{member} {'present' if member in l1 else 'missing'}",
                      level="L1", severity="warning")
        if "probes" in l1:
            probes = pack.json(l1["probes"]["file"])
            ps = ref.l1()["probe_structure"]
            ok = int(probes["n_atoms"]) == int(ps["n_atoms"]) and np.isclose(probes["volume_A3"], ps["volume_A3"])
            check("L1.probe_structure", ok, "probe measured on the reference probe cell" if ok
                  else "probe cell differs from the reference", level="L1")

    l2 = pack.level("L2")
    if l2:
        ref_l2 = ref.l2()
        relax = pack.json(l2["relax"]["file"])
        for label, rec in relax["relaxations"].items():
            want_comp = ref_l2["compositions"].get(_relax_label_key(label))
            check(f"L2.relax.{label}.composition", want_comp == rec.get("composition"),
                  f"composition {rec.get('composition')} vs reference {want_comp}", level="L2")
        kind = l2.get("eos_scale_kind", "linear")
        check("L2.eos.scale_kind", kind == "volume",
              "EOS scanned in volume, like the DFT tasks" if kind == "volume" else
              "EOS scanned with linear scale factors (V/V0 0.78-1.26) against a DFT window of "
              "V/V0 0.92-1.08: V0/B0 errors include fit-range bias; rerun L2 to fix",
              level="L2", severity="warning")

    errors = [c for c in checks if not c["ok"] and c["severity"] == "error"]
    warnings = [c for c in checks if not c["ok"] and c["severity"] == "warning"]
    blocked = sorted({c["level"] for c in errors if c["level"] != "-"})
    return {
        "ok": not errors,
        "pack_id": pack.pack_id,
        "model_key": pack.model_key,
        "reference_id": ref.reference_id,
        "blocked_levels": blocked if all(c["level"] != "-" for c in errors) else ["L0", "L1", "L2"],
        "n_errors": len(errors),
        "n_warnings": len(warnings),
        "checks": checks,
    }
