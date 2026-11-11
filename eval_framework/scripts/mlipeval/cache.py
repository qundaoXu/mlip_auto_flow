"""Tiny npz-backed cache for expensive model evaluations.

Notebooks must be re-runnable without paying for a 615-frame MACE evaluation
again, so every driver stores its :class:`~mlipeval.calculators.Predictions` in
``eval/data/cache`` keyed by (model checkpoint hash, frame-set fingerprint).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path  # noqa: F401  (used in a type hint below)
from typing import Sequence

import numpy as np
from ase import Atoms

from .calculators import Model, Predictions, descriptors, predict
from .schema import ModelSpec, as_spec
from .paths import data_path


def frames_fingerprint(frames: Sequence[Atoms]) -> str:
    h = hashlib.sha256()
    h.update(str(len(frames)).encode())
    for atoms in frames[:: max(1, len(frames) // 32)]:
        h.update(np.asarray(atoms.positions, dtype=np.float32).tobytes())
        h.update(np.asarray(atoms.cell.array, dtype=np.float32).tobytes())
        h.update("".join(atoms.get_chemical_symbols()).encode())
    return h.hexdigest()[:16]


_spec = as_spec


def cache_key(model: "Model | ModelSpec", frames: Sequence[Atoms], tag: str) -> str:
    """Cache key: tag + checkpoint + frame set + device/dtype.

    ``device`` and ``dtype`` belong in the key because CPU and GPU kernels do not
    produce bit-identical results; without them a GPU job would silently reuse
    numbers computed somewhere else, and the provenance recorded in the result
    file would be wrong.
    """
    spec = _spec(model)
    # ``artifact`` distinguishes an eager checkpoint from a frozen/compiled export
    # of the same weights; they are not guaranteed to agree numerically.
    artifact = spec.software.get("artifact", "")
    suffix = f"__{artifact}" if artifact else ""
    return (
        f"{tag}__{spec.checkpoint_sha256[:10]}__{spec.device}"
        f"_{spec.dtype}{suffix}__{frames_fingerprint(frames)}"
    )


def _save(path: Path, predictions: Predictions) -> None:
    np.savez_compressed(
        path,
        energies=predictions.energies,
        forces=np.concatenate([f for f in predictions.forces]) if predictions.forces else np.zeros((0, 3)),
        force_splits=np.cumsum([len(f) for f in predictions.forces])[:-1],
        stresses=predictions.stresses if predictions.stresses is not None else np.zeros((0, 3, 3)),
        n_atoms=predictions.n_atoms,
        volumes=predictions.volumes,
        wall_time_s=predictions.wall_time_s,
        failures=json.dumps(predictions.failures),
        has_stress=predictions.stresses is not None,
    )


def _load(path: Path) -> Predictions:
    z = np.load(path, allow_pickle=False)
    forces = np.split(z["forces"], z["force_splits"]) if z["forces"].size else []
    return Predictions(
        energies=z["energies"],
        forces=[np.asarray(f) for f in forces],
        stresses=z["stresses"] if bool(z["has_stress"]) else None,
        n_atoms=z["n_atoms"],
        volumes=z["volumes"],
        wall_time_s=float(z["wall_time_s"]),
        failures=json.loads(str(z["failures"])),
    )


def load_predictions(path: Path) -> Predictions:
    """Read a prediction cache file (public name for evaluation packs)."""
    return _load(Path(path))


def _any_device_match(frames: Sequence[Atoms], model: Model, tag: str) -> Path | None:
    """A cache entry for the same model and frames but a different device.

    CPU and GPU MACE in ``float64`` were verified to agree bit-for-bit on this
    workload (615 frames: identical energy MAE 14.677986 meV/atom and force MAE
    0.103261 eV/A), so reusing across devices is sound.  It matters because a
    notebook opened on the login node would otherwise recompute everything a
    GPU job already produced.

    Crossing *devices* is sound; crossing *artefacts* is not.  A frozen or
    compiled export of the same weights is a different computation -- DPA-4's
    AOTInductor archive cannot even evaluate a dimer that its eager checkpoint
    handles fine -- so the artefact is required to match here exactly as it does
    in :func:`cache_key`.  Entries written before the artefact joined the key
    carry no artefact token and are only reused by a model that names none.

    Two layouts are recognised: ``<prefix><device>_<dtype>[__<artifact>]__<fp>.npz``
    (current) and ``<prefix><fp>.npz`` (written before the device joined the key).
    """
    spec = _spec(model)
    prefix = f"{tag}__{spec.checkpoint_sha256[:10]}__"
    artifact = spec.software.get("artifact", "")
    tail = f"{('__' + artifact) if artifact else ''}__{frames_fingerprint(frames)}.npz"
    directory = data_path("cache", "_").parent

    same_dtype = sorted(directory.glob(f"{prefix}*_{spec.dtype}{tail}"))
    if same_dtype:
        return same_dtype[0]
    any_device = sorted(directory.glob(f"{prefix}*{tail}"))
    if any_device:
        return any_device[0]
    if artifact:
        return None  # never fall back to an entry whose artefact is unknown
    legacy = directory / f"{prefix}{frames_fingerprint(frames)}.npz"
    return legacy if legacy.exists() else None


def cached_predict(
    frames: Sequence[Atoms],
    model: Model,
    *,
    tag: str,
    with_stress: bool = True,
    refresh: bool = False,
    cross_device: bool = True,
    progress=None,
) -> Predictions:
    """Evaluate ``frames`` once; reuse the npz on every later call.

    ``cross_device`` lets a run on one device reuse a cache written on another.
    Without it, opening a notebook on the login node (CPU) would recompute the
    615-frame evaluation that a GPU job had already done -- minutes of work on a
    shared head node, for numbers that are bit-identical.
    """
    path = data_path("cache", f"{cache_key(model, frames, tag)}.npz")
    if path.exists() and not refresh:
        return _load(path)
    if cross_device and not refresh:
        other = _any_device_match(frames, model, tag)
        if other is not None and other.exists():
            print(f"[cache] reusing {other.name}\n"
                  f"        (written on a different device; float64 CPU/GPU agree bit-for-bit here)")
            return _load(other)
    predictions = predict(frames, model, with_stress=with_stress, progress=progress)
    _save(path, predictions)
    return predictions


def cached_descriptors(
    frames: Sequence[Atoms],
    model: Model,
    *,
    tag: str,
    invariants_only: bool = True,
    refresh: bool = False,
    cross_device: bool = True,
) -> np.ndarray:
    """Per-structure MACE descriptors, cached like predictions.

    Computing these is a full forward pass per frame -- the same cost as an
    energy evaluation -- so anything that selects or de-duplicates structures
    over a large frame set has to reuse them rather than recompute.
    """
    path = data_path("cache", f"desc_{cache_key(model, frames, tag)}.npz")
    if path.exists() and not refresh:
        return np.load(path)["descriptors"]
    if cross_device and not refresh:
        other = _any_device_match(frames, model, f"desc_{tag}")
        if other is not None and other.exists():
            return np.load(other)["descriptors"]
    if not hasattr(model, "calculator"):
        raise RuntimeError(
            f"no cached descriptors for tag '{tag}' and only a ModelSpec was given. "
            "Populate the cache from a PBS job (`submit_pbs.py --command \"check_notebooks.py\"`) "
            "instead of evaluating the model here."
        )
    values = descriptors(frames, model, invariants_only=invariants_only)
    np.savez_compressed(path, descriptors=values)
    return values


def cached_binary_pecs(
    model: Model,
    species: Sequence[str],
    *,
    r_min: float = 1.0,
    r_max: float = 8.0,
    dr: float = 0.02,
    refresh: bool = False,
) -> dict:
    """Binary potential energy curves, cached to ``eval/data/cache``.

    Six curves at 0.02 A spacing are ~2100 dimer evaluations.  That is a couple
    of minutes on a login-node CPU and is recomputed on every notebook re-run
    unless it is cached, so the cache is what keeps nb02 readable interactively
    while the numbers still come from a PBS job.
    """
    from .l1_pes import all_binary_pecs

    spec = _spec(model)
    # v2: PECResult gained a measured per-curve noise floor, so v1 entries would
    # silently make the continuity test behave as if the model were noiseless.
    key = f"pec2__{spec.checkpoint_sha256[:10]}__{spec.software.get('artifact', '')}" \
          f"__{'-'.join(sorted(species))}__{r_min:g}_{r_max:g}_{dr:g}"
    path = data_path("cache", f"{key}__{spec.device}_{spec.dtype}.npz")
    if not path.exists() and not refresh:
        alternatives = sorted(path.parent.glob(f"{key}__*.npz"))
        if alternatives:
            path = alternatives[0]

    if path.exists() and not refresh:
        # Only the curve itself and the noise probe are restored: those cost
        # ~350 model evaluations per pair plus five more for the probe.  Every
        # judgement about the curve is recomputed by ``diagnose_pec``, so a
        # change to a continuity criterion takes effect on cached curves instead
        # of being masked by a stored verdict.
        out, stale = pecs_from_npz(np.load(path, allow_pickle=False))
        if stale:
            # Keep the file self-describing for readers that plot straight from
            # the npz (nb07 does) rather than leaving two answers on disk.
            _write_pec_npz(path, out)
        return out

    if not hasattr(model, "calculator"):
        raise RuntimeError(
            "no cached PEC for this checkpoint and grid, and only a ModelSpec was given. "
            "Populate the cache from a PBS job (`submit_pbs.py --command \"run_l1.py\"`) "
            "rather than computing 2100 dimers here."
        )
    pecs = all_binary_pecs(model, species, r_min=r_min, r_max=r_max, dr=dr)
    _write_pec_npz(data_path("cache", f"{key}__{spec.device}_{spec.dtype}.npz"), pecs)
    return pecs


def pecs_from_npz(z) -> tuple[dict, bool]:
    """Rebuild :class:`~mlipeval.l1_pes.PECResult` objects from a stored PEC file.

    Only the curve and the measured noise are trusted; every verdict is
    recomputed by ``diagnose_pec``.  Returns the curves and whether the stored
    derived columns disagree with the recomputation (a stale file).
    """
    from .l1_pes import diagnose_pec

    pairs = [str(k) for k in z["pairs"]]
    out = {}
    stale = False
    for i, pair in enumerate(pairs):
        out[pair] = diagnose_pec(
            pair,
            z["r"],
            z["energy"][i],
            z["force_analytic"][i],
            noise_residual=float(z["noise_residual"][i]) if "noise_residual" in z else 0.0,
            error=(str(z["error"][i]) or None) if "error" in z else None,
        )
        if "convergence_ratio" in z and np.isfinite(z["convergence_ratio"][i]):
            stale |= not np.isclose(z["convergence_ratio"][i],
                                    out[pair].convergence_ratio, rtol=1e-6)
    return out, stale


def _write_pec_npz(path, pecs: dict) -> None:
    """Serialise the curves.  The derived columns are written for readers that
    plot straight from the npz; :func:`cached_binary_pecs` never trusts them on
    the way back in."""
    pairs = sorted(pecs)
    np.savez_compressed(
        path,
        pairs=np.array(pairs),
        r=pecs[pairs[0]].r,
        energy=np.array([pecs[p].energy for p in pairs]),
        force_analytic=np.array([pecs[p].force_analytic for p in pairs]),
        force_numeric=np.array([pecs[p].force_numeric for p in pairs]),
        r_min=np.array([pecs[p].r_min for p in pairs]),
        e_min=np.array([pecs[p].e_min for p in pairs]),
        monotonic_repulsive=np.array([pecs[p].monotonic_repulsive for p in pairs]),
        max_jump_ev=np.array([pecs[p].max_jump_ev for p in pairs]),
        max_force_mismatch=np.array([pecs[p].max_force_mismatch for p in pairs]),
        convergence_ratio=np.array([pecs[p].convergence_ratio for p in pairs]),
        n_nonfinite=np.array([pecs[p].n_nonfinite for p in pairs]),
        noise_residual=np.array([pecs[p].noise_residual for p in pairs]),
        error=np.array([pecs[p].error or "" for p in pairs]),
    )
