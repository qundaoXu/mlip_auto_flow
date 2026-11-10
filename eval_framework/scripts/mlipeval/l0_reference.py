"""L0 -- numerical consistency against the reference labels.

Scope (``validation_methods_L0-L5.md`` L0):

1. energy / force / stress MAE, RMSE, P99, max;
2. energies compared only after removing a per-composition constant offset;
3. force errors binned by species, force magnitude, temperature stage and
   configuration type;
4. worst frames and worst atoms exported with their structures;
5. reference protocol, model native protocol and leakage risk recorded.

Nothing here silently pools protocols: :func:`evaluate_l0` takes exactly one
:class:`~mlipeval.io_labeled.LabelledSet`, whose ``protocol_id`` is stamped onto
every metric it produces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from ase import Atoms

from .calculators import Model, Predictions, predict
from .io_labeled import LabelledSet, VirialAudit, audit_virial_convention, virial_to_stress
from .schema import EvidenceTier, MetricResult, ModelSpec, ResultSet, Status, as_spec


# --------------------------------------------------------------------------
# error statistics
# --------------------------------------------------------------------------
def error_stats(errors: np.ndarray) -> dict[str, float]:
    """MAE / RMSE / P99 / max / bias over a flat error array."""
    e = np.asarray(errors, dtype=float).reshape(-1)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return {k: float("nan") for k in ("mae", "rmse", "p99", "max", "bias", "n")}
    a = np.abs(e)
    return {
        "mae": float(a.mean()),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
        "bias": float(e.mean()),
        "n": int(e.size),
    }


def constant_energy_offset(
    ref_energy: np.ndarray,
    model_energy: np.ndarray,
    n_atoms: np.ndarray,
    compositions: np.ndarray,
) -> dict[str, float]:
    """Per-composition constant shift, fitted as a per-atom offset.

    ``database_link.md`` 3.4 allows removing a constant energy offset *within*
    one composition and one protocol, and nothing else.  The offset is defined
    so that ``E_model - n_atoms * offset`` matches ``E_ref`` in the mean.
    """
    offsets: dict[str, float] = {}
    for comp in np.unique(compositions):
        m = compositions == comp
        delta = model_energy[m] - ref_energy[m]
        ok = np.isfinite(delta)
        offsets[str(comp)] = float(np.sum(delta[ok]) / np.sum(n_atoms[m][ok])) if ok.any() else float("nan")
    return offsets


def apply_offset(
    model_energy: np.ndarray,
    n_atoms: np.ndarray,
    compositions: np.ndarray,
    offsets: Mapping[str, float],
) -> np.ndarray:
    shift = np.array([offsets.get(str(c), 0.0) for c in compositions])
    return model_energy - shift * n_atoms


# --------------------------------------------------------------------------
# binning
# --------------------------------------------------------------------------
def bin_force_errors_by_species(
    ref_forces: Sequence[np.ndarray],
    model_forces: Sequence[np.ndarray],
    symbols: Sequence[np.ndarray],
) -> dict[str, np.ndarray]:
    """Flat per-component force errors grouped by chemical species."""
    buckets: dict[str, list[np.ndarray]] = {}
    for fr, fm, sym in zip(ref_forces, model_forces, symbols):
        d = (fm - fr).reshape(len(sym), 3)
        for s in np.unique(sym):
            buckets.setdefault(str(s), []).append(d[sym == s].reshape(-1))
    return {k: np.concatenate(v) for k, v in buckets.items()}


def bin_force_errors_by_magnitude(
    ref_forces: Sequence[np.ndarray],
    model_forces: Sequence[np.ndarray],
    edges: Sequence[float] = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, np.inf),
) -> dict[str, np.ndarray]:
    """Per-atom force-vector errors grouped by reference force magnitude."""
    edges = list(edges)
    buckets: dict[str, list[np.ndarray]] = {}
    for fr, fm in zip(ref_forces, model_forces):
        mag = np.linalg.norm(fr, axis=1)
        err = np.linalg.norm(fm - fr, axis=1)
        idx = np.digitize(mag, edges[1:-1])
        for b in np.unique(idx):
            lo, hi = edges[b], edges[b + 1]
            label = f"[{lo:g},{hi:g})" if np.isfinite(hi) else f">={lo:g}"
            buckets.setdefault(label, []).append(err[idx == b])
    return {k: np.concatenate(v) for k, v in sorted(buckets.items())}


def bin_errors_by_tag(errors_per_frame: np.ndarray, tag: np.ndarray) -> dict[str, np.ndarray]:
    """Group per-frame quantities by an arbitrary categorical tag."""
    out: dict[str, np.ndarray] = {}
    for value in np.unique(tag):
        out[str(value)] = np.asarray(errors_per_frame)[tag == value]
    return out


# --------------------------------------------------------------------------
# worst-case export
# --------------------------------------------------------------------------
@dataclass
class WorstCases:
    """Highest-error frames and atoms, kept together with their structures."""

    frame_indices: np.ndarray
    frame_scores: np.ndarray
    frame_reason: str
    atom_frames: np.ndarray
    atom_indices: np.ndarray
    atom_scores: np.ndarray
    atom_species: np.ndarray
    structures: list[Atoms] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_reason": self.frame_reason,
            "worst_frames": [
                {"frame": int(i), "score": float(s)} for i, s in zip(self.frame_indices, self.frame_scores)
            ],
            "worst_atoms": [
                {"frame": int(f), "atom": int(a), "species": str(sp), "force_error_eV_per_A": float(s)}
                for f, a, sp, s in zip(self.atom_frames, self.atom_indices, self.atom_species, self.atom_scores)
            ],
        }


def worst_cases(
    labelled: LabelledSet,
    predictions: Predictions,
    *,
    n_frames: int = 10,
    n_atoms: int = 25,
) -> WorstCases:
    ref_forces = labelled.forces()
    per_frame = np.array(
        [
            np.nanmax(np.linalg.norm(fm - fr, axis=1)) if np.isfinite(fm).all() else np.inf
            for fr, fm in zip(ref_forces, predictions.forces)
        ]
    )
    order = np.argsort(-np.nan_to_num(per_frame, nan=-1.0, posinf=1e30))[:n_frames]

    all_err, all_frame, all_atom, all_sym = [], [], [], []
    symbols = labelled.symbols()
    for k, (fr, fm) in enumerate(zip(ref_forces, predictions.forces)):
        err = np.linalg.norm(fm - fr, axis=1)
        all_err.append(err)
        all_frame.append(np.full(err.size, k))
        all_atom.append(np.arange(err.size))
        all_sym.append(symbols[k])
    all_err = np.concatenate(all_err)
    all_frame = np.concatenate(all_frame)
    all_atom = np.concatenate(all_atom)
    all_sym = np.concatenate(all_sym)
    atom_order = np.argsort(-np.nan_to_num(all_err, nan=-1.0))[:n_atoms]

    return WorstCases(
        frame_indices=order,
        frame_scores=per_frame[order],
        frame_reason="max per-atom force-vector error in the frame",
        atom_frames=all_frame[atom_order],
        atom_indices=all_atom[atom_order],
        atom_scores=all_err[atom_order],
        atom_species=all_sym[atom_order],
        structures=[labelled.frames[i].copy() for i in order],
    )


# --------------------------------------------------------------------------
# top-level driver
# --------------------------------------------------------------------------
@dataclass
class L0Report:
    """Everything L0 produced, kept so the notebook can plot without recompute."""

    results: ResultSet
    predictions: Predictions
    offsets: dict[str, float]
    energy_ref: np.ndarray
    energy_model_shifted: np.ndarray
    energy_error_per_atom: np.ndarray
    force_error_by_species: dict[str, np.ndarray]
    force_error_by_magnitude: dict[str, np.ndarray]
    force_error_by_tag: dict[str, dict[str, np.ndarray]]
    force_ref_flat: np.ndarray
    force_model_flat: np.ndarray
    worst: WorstCases
    virial_audit: VirialAudit
    stress_ref: np.ndarray | None
    stress_model: np.ndarray | None


def evaluate_l0(
    labelled: LabelledSet,
    model: "Model | ModelSpec",
    *,
    predictions: Predictions | None = None,
    reference_protocol_id: str | None = None,
    tags: Mapping[str, np.ndarray] | None = None,
    force_magnitude_edges: Sequence[float] = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, np.inf),
    leakage_note: str | None = None,
    progress=None,
) -> L0Report:
    """Run the full L0 battery on one labelled set with one model.

    ``predictions`` may be supplied to reuse an earlier expensive evaluation.
    A set with a ``stress_key`` (harvested from OUTCAR, ASE sign) is scored on
    its stress directly; for a bare ``Virial`` the stress stays a *diagnostic*
    unless the virial convention audit resolved, matching the L0 precondition
    in the framework document.
    """
    ref_protocol = reference_protocol_id or labelled.protocol_id
    spec = as_spec(model)
    model_protocol_id = f"{spec.name}:{spec.checkpoint_sha256[:12]}"

    if predictions is None:
        if not hasattr(model, "calculator"):
            raise RuntimeError(
                "evaluate_l0 was given a ModelSpec but no predictions; either pass "
                "cached predictions, or hand it a loaded Model from a PBS job."
            )
        predictions = predict(labelled.frames, model, with_stress=True, progress=progress)

    results = ResultSet(
        "L0",
        model=spec,
        reference_protocol=ref_protocol,
        reference_source=labelled.source,
        reference_role=labelled.role.value,
        n_frames=len(labelled),
        protocol_mismatch=(
            f"reference '{ref_protocol}' vs model native '{spec.native_protocol}' -- "
            "different theory levels; energies are only compared after per-composition offset removal"
        ),
        leakage_note=leakage_note or "not screened",
    )

    def add(metric_id, value, unit, **kw) -> MetricResult:
        return results.add(
            MetricResult(
                metric_id=metric_id,
                level="L0",
                value=value,
                unit=unit,
                reference_protocol=ref_protocol,
                model_protocol=model_protocol_id,
                **kw,
            )
        )

    # -- model health ------------------------------------------------------
    n_fail = len(predictions.failures)
    add(
        "L0.model_eval_failures",
        n_fail,
        "count",
        status=Status.FAILED if n_fail else Status.OK,
        evidence_tier=EvidenceTier.S,
        failure_reason=None if not n_fail else str(predictions.failures[:5]),
    )
    n_nonfinite = int(np.sum(~np.isfinite(predictions.energies)))
    add(
        "L0.nonfinite_energies",
        n_nonfinite,
        "count",
        status=Status.FAILED if n_nonfinite else Status.OK,
        evidence_tier=EvidenceTier.S,
    )

    # -- energies ----------------------------------------------------------
    e_ref = labelled.energies
    n_at = labelled.n_atoms
    comps = labelled.compositions()
    offsets = constant_energy_offset(e_ref, predictions.energies, n_at, comps)
    e_model = apply_offset(predictions.energies, n_at, comps, offsets)
    e_err_pa = (e_model - e_ref) / n_at

    stats_e = error_stats(e_err_pa * 1000.0)  # meV/atom
    for key in ("mae", "rmse", "p99", "max"):
        add(
            f"L0.energy_{key}",
            stats_e[key],
            "meV/atom",
            evidence_tier=EvidenceTier.B,
            status=Status.PROVISIONAL,
            extra={
                "note": "per-composition constant offset removed before comparison",
                "offsets_eV_per_atom": offsets,
            },
        )
    add(
        "L0.energy_offset_spread",
        float(np.ptp(list(offsets.values()))) * 1000.0 if len(offsets) > 1 else 0.0,
        "meV/atom",
        evidence_tier=EvidenceTier.S,
        status=Status.OK,
        extra={"offsets_eV_per_atom": offsets, "n_compositions": len(offsets)},
    )

    # -- forces ------------------------------------------------------------
    ref_forces = labelled.forces()
    f_ref_flat = np.concatenate([f.reshape(-1) for f in ref_forces])
    f_model_flat = np.concatenate([f.reshape(-1) for f in predictions.forces])
    stats_f = error_stats(f_model_flat - f_ref_flat)
    for key in ("mae", "rmse", "p99", "max"):
        add(f"L0.force_{key}", stats_f[key], "eV/A", evidence_tier=EvidenceTier.B, status=Status.PROVISIONAL)
    add(
        "L0.force_component_r2",
        float(np.corrcoef(f_ref_flat, f_model_flat)[0, 1] ** 2),
        "-",
        evidence_tier=EvidenceTier.B,
        status=Status.OK,
    )

    symbols = labelled.symbols()
    by_species = bin_force_errors_by_species(ref_forces, predictions.forces, symbols)
    for species, err in sorted(by_species.items()):
        s = error_stats(err)
        add(
            f"L0.force_mae.species={species}",
            s["mae"],
            "eV/A",
            evidence_tier=EvidenceTier.B,
            status=Status.PROVISIONAL,
            extra={"rmse": s["rmse"], "p99": s["p99"], "max": s["max"], "n": s["n"]},
        )

    by_magnitude = bin_force_errors_by_magnitude(ref_forces, predictions.forces, force_magnitude_edges)
    for label, err in by_magnitude.items():
        s = error_stats(err)
        add(
            f"L0.force_vector_mae.|F|={label}",
            s["mae"],
            "eV/A",
            evidence_tier=EvidenceTier.B,
            status=Status.PROVISIONAL,
            extra={"p99": s["p99"], "max": s["max"], "n": s["n"]},
        )

    # per-frame force RMSE, then grouped by any categorical tag supplied
    per_frame_frmse = np.array(
        [float(np.sqrt(np.mean((fm - fr) ** 2))) for fr, fm in zip(ref_forces, predictions.forces)]
    )
    by_tag: dict[str, dict[str, np.ndarray]] = {}
    for tag_name, tag_values in (tags or {}).items():
        tag_values = np.asarray(tag_values)
        if tag_values.dtype.kind in "fc" or tag_values.shape[0] != len(labelled):
            continue
        groups = bin_errors_by_tag(per_frame_frmse, tag_values)
        by_tag[tag_name] = groups
        for group, values in sorted(groups.items()):
            add(
                f"L0.force_rmse.{tag_name}={group}",
                float(np.mean(values)),
                "eV/A",
                evidence_tier=EvidenceTier.B,
                status=Status.PROVISIONAL,
                extra={"n_frames": int(values.size), "max_frame_rmse": float(values.max())},
            )
        # the offset is fitted over the whole set, so a stage that sits off the
        # common offset shows up here rather than being absorbed by it
        for group, values in sorted(bin_errors_by_tag(np.abs(e_err_pa) * 1000.0, tag_values).items()):
            add(
                f"L0.energy_mae.{tag_name}={group}",
                float(np.nanmean(values)),
                "meV/atom",
                evidence_tier=EvidenceTier.B,
                status=Status.PROVISIONAL,
                extra={"n_frames": int(values.size), "note": "after the set-wide per-composition offset"},
            )

    # -- stress ------------------------------------------------------------
    audit = audit_virial_convention(labelled, predictions.stresses)
    stress_ref = stress_model = None
    virials = labelled.virials()
    known = labelled.stresses()
    if known is not None and predictions.stresses is not None:
        # harvested from OUTCAR in the ASE convention, so there is no sign to audit
        mask, stress_ref = known
        stress_model = predictions.stresses[mask]
        stats_s = error_stats((stress_model - stress_ref).reshape(-1) * 1000.0)  # meV/A^3
        for key in ("mae", "rmse", "max"):
            add(
                f"L0.stress_{key}",
                stats_s[key],
                "meV/A^3",
                evidence_tier=EvidenceTier.B,
                status=Status.PROVISIONAL,
                extra={
                    "n_frames": int(mask.sum()),
                    "n_frames_without_stress": int((~mask).sum()),
                    "convention": f"{labelled.stress_key} read from OUTCAR, ASE sign; no virial audit needed",
                },
            )
    elif virials is not None and predictions.stresses is not None:
        sign = +1 if audit.recommendation.startswith("stress = +") else -1
        stress_ref = virial_to_stress(virials, labelled.volumes, sign=sign)
        stress_model = predictions.stresses
        stats_s = error_stats((stress_model - stress_ref).reshape(-1) * 1000.0)  # meV/A^3
        resolved = audit.recommendation != "undetermined"
        add(
            "L0.stress_rmse",
            stats_s["rmse"],
            "meV/A^3",
            evidence_tier=EvidenceTier.C,
            status=Status.PROVISIONAL if resolved else Status.SKIPPED,
            failure_reason=None if resolved else audit.verdict,
            extra={
                "virial_audit": audit.to_dict(),
                "assumed_convention": audit.recommendation,
                "warning": "reference virial protocol is not traceable; diagnostic only, never scored",
            },
        )
    else:
        add(
            "L0.stress_rmse",
            None,
            "meV/A^3",
            status=Status.SKIPPED,
            evidence_tier=EvidenceTier.C,
            failure_reason=audit.verdict,
        )

    worst = worst_cases(labelled, predictions)
    add(
        "L0.worst_cases",
        worst.to_dict(),
        "-",
        status=Status.OK,
        evidence_tier=EvidenceTier.S,
        extra={"n_frames": len(worst.frame_indices), "n_atoms": len(worst.atom_indices)},
    )

    return L0Report(
        results=results,
        predictions=predictions,
        offsets=offsets,
        energy_ref=e_ref,
        energy_model_shifted=e_model,
        energy_error_per_atom=e_err_pa,
        force_error_by_species=by_species,
        force_error_by_magnitude=by_magnitude,
        force_error_by_tag=by_tag,
        force_ref_flat=f_ref_flat,
        force_model_flat=f_model_flat,
        worst=worst,
        virial_audit=audit,
        stress_ref=stress_ref,
        stress_model=stress_model,
    )
