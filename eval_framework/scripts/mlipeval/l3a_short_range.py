"""L3a -- short-range structure and chemical order of the disordered phases.

Scope (``validation_methods_L0-L5.md`` L3a):

1. six partial RDFs with peak position, height, width and first minimum;
2. species-resolved angular distribution functions;
3. full Ge/Sb/Te coordination-number distributions;
4. homopolar bond fraction;
5. Ge tetrahedral order-parameter distribution;
6. 1-Wasserstein and Jensen-Shannon distances between reference and model.

The controlling rule is that every post-processing knob is frozen once, on the
AIMD side, and reused verbatim for the MLIP trajectory.  That is what
:func:`freeze_params_from_reference` produces and what every function here
consumes; none of them derives a cutoff on its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io_vasp import Trajectory
from .neighbors import (
    StructureParams,
    all_pairs,
    bond_angle_distribution,
    coordination_distribution,
    derive_bond_cutoffs,
    first_minimum,
    first_peak,
    homopolar_fraction,
    pair_key,
    partial_rdf,
    tetrahedral_order,
)
from .schema import EvidenceTier, MetricResult, ResultSet, Status


def _result(metric_id: str, value, unit: str, **kw) -> MetricResult:
    kw.setdefault("evidence_tier", EvidenceTier.A)
    kw.setdefault("status", Status.PROVISIONAL)
    return MetricResult(metric_id=metric_id, level="L3a", value=value, unit=unit, **kw)


# --------------------------------------------------------------------------
# freezing the analysis parameters
# --------------------------------------------------------------------------
def freeze_params_from_reference(
    reference: Trajectory,
    *,
    stride: int = 1,
    r_max: float = 10.0,
    n_bins: int = 400,
    cutoff_search: tuple[float, float] = (1.8, 4.5),
    peak_rule: str = "max",
    **overrides,
) -> tuple[StructureParams, dict[str, np.ndarray]]:
    """Derive bond cutoffs from the *reference* partial RDF first minima.

    Returns the frozen :class:`StructureParams` together with the reference RDF
    it was derived from, so the notebook can show exactly where each cutoff came
    from.  ``params.cutoff_diagnostics`` records, per pair, the peak, the minimum,
    how deep that minimum is and whether the value was measured or inherited from
    the total RDF.  Apply the returned object unchanged to every other trajectory.
    """
    seed = StructureParams(r_max=r_max, n_bins=n_bins, provenance="pending")
    rdf = partial_rdf(reference, seed, stride=stride)
    cutoffs, diagnostics = derive_bond_cutoffs(
        rdf,
        reference.species,
        reference.symbols,
        r_lo=cutoff_search[0],
        r_hi=cutoff_search[1],
        peak_rule=peak_rule,
    )
    params = StructureParams(
        r_max=r_max,
        n_bins=n_bins,
        bond_cutoffs=cutoffs,
        provenance=(
            f"first minimum of the partial RDF of '{reference.name}' "
            f"(protocol_id={reference.protocol.protocol_id if reference.protocol else 'unknown'}, "
            f"{len(reference)} frames, stride={stride}, search window {cutoff_search[0]}-{cutoff_search[1]} A"
            + (f", peak_rule={peak_rule}" if peak_rule != "max" else "")
            + ")"
        ),
        cutoff_diagnostics=diagnostics,
        **overrides,
    )
    return params, rdf


# --------------------------------------------------------------------------
# distribution distances
# --------------------------------------------------------------------------
def wasserstein1(x: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
    """1-Wasserstein distance between two densities on a shared grid ``x``."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    ok = np.isfinite(p) & np.isfinite(q)
    x, p, q = np.asarray(x)[ok], p[ok], q[ok]
    if p.sum() <= 0 or q.sum() <= 0:
        return float("nan")
    cp = np.cumsum(p) / p.sum()
    cq = np.cumsum(q) / q.sum()
    return float(np.trapezoid(np.abs(cp - cq), x))


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon distance (base 2, so it lies in ``[0, 1]``)."""
    from scipy.spatial.distance import jensenshannon

    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    ok = np.isfinite(p) & np.isfinite(q)
    p, q = np.clip(p[ok], 0, None), np.clip(q[ok], 0, None)
    if p.sum() <= 0 or q.sum() <= 0:
        return float("nan")
    return float(jensenshannon(p / p.sum(), q / q.sum(), base=2))


def sample_distances(reference_samples: np.ndarray, model_samples: np.ndarray, *, bins: int = 60) -> dict[str, float]:
    """Distances between two raw sample sets (used for CN / tetrahedral q)."""
    from scipy.stats import wasserstein_distance

    reference_samples = np.asarray(reference_samples, dtype=float)
    model_samples = np.asarray(model_samples, dtype=float)
    if reference_samples.size == 0 or model_samples.size == 0:
        return {"wasserstein1": float("nan"), "jensen_shannon": float("nan")}
    lo = float(min(reference_samples.min(), model_samples.min()))
    hi = float(max(reference_samples.max(), model_samples.max()))
    edges = np.linspace(lo, hi if hi > lo else lo + 1.0, bins + 1)
    p, _ = np.histogram(reference_samples, bins=edges, density=True)
    q, _ = np.histogram(model_samples, bins=edges, density=True)
    return {
        "wasserstein1": float(wasserstein_distance(reference_samples, model_samples)),
        "jensen_shannon": jensen_shannon(p, q),
    }


# --------------------------------------------------------------------------
# the L3a battery
# --------------------------------------------------------------------------
@dataclass
class L3aReport:
    params: StructureParams
    rdf_ref: dict[str, np.ndarray]
    rdf_model: dict[str, np.ndarray] | None
    adf_ref: dict[str, np.ndarray]
    adf_model: dict[str, np.ndarray] | None
    cn_ref: dict[str, np.ndarray]
    cn_model: dict[str, np.ndarray] | None
    homopolar_ref: dict[str, float]
    homopolar_model: dict[str, float] | None
    tetra_ref: np.ndarray
    tetra_model: np.ndarray | None
    peaks_ref: dict[str, dict[str, float]]
    peaks_model: dict[str, dict[str, float]] | None
    results: ResultSet


def analyse_short_range(
    reference: Trajectory,
    params: StructureParams,
    *,
    model_traj: Trajectory | None = None,
    stride: int = 1,
    model_stride: int | None = None,
    tetrahedral_species: str = "Ge",
    state_label: str = "state",
    n_independent_references: int = 1,
) -> L3aReport:
    """Run the whole L3a battery on one reference trajectory (+ optional model).

    Every metric is emitted for the reference alone as well, so the notebook can
    show the reference value even when no MLIP trajectory exists yet.
    """
    model_stride = stride if model_stride is None else model_stride
    results = ResultSet(
        "L3a",
        state=state_label,
        frozen_params=params.to_dict(),
        reference=reference.describe(),
        model_trajectory=model_traj.describe() if model_traj else None,
        n_independent_references=n_independent_references,
    )
    tier = EvidenceTier.A if model_traj is not None else EvidenceTier.S
    ref_pid = reference.protocol.protocol_id if reference.protocol else None
    mod_pid = model_traj.protocol.protocol_id if (model_traj and model_traj.protocol) else None

    def add(metric_id, value, unit, **kw):
        kw.setdefault("evidence_tier", tier)
        kw.setdefault("reference_protocol", ref_pid)
        kw.setdefault("model_protocol", mod_pid)
        return results.add(
            MetricResult(metric_id=f"{metric_id}@{state_label}", level="L3a", value=value, unit=unit, **kw)
        )

    # -- 1. partial RDFs ---------------------------------------------------
    rdf_ref = partial_rdf(reference, params, stride=stride)
    rdf_model = partial_rdf(model_traj, params, stride=model_stride) if model_traj else None
    r = rdf_ref["r"]

    peaks_ref: dict[str, dict[str, float]] = {}
    peaks_model: dict[str, dict[str, float]] | None = {} if rdf_model else None
    for a, b in all_pairs(reference.species):
        key = pair_key(a, b)
        pr = first_peak(r, rdf_ref[key])
        pr["first_minimum"] = first_minimum(r, rdf_ref[key])["r_min"]
        peaks_ref[key] = pr
        if rdf_model is not None:
            pm = first_peak(r, rdf_model[key])
            pm["first_minimum"] = first_minimum(r, rdf_model[key])["r_min"]
            peaks_model[key] = pm
            for field_name, unit in (("r_peak", "A"), ("g_peak", "-"), ("fwhm", "A")):
                add(
                    f"L3a.rdf.{key}.{field_name}",
                    pm[field_name],
                    unit,
                    status=Status.PROVISIONAL,
                    extra={
                        "reference": pr[field_name],
                        "model": pm[field_name],
                        "difference": pm[field_name] - pr[field_name],
                    },
                )
            add(
                f"L3a.rdf.{key}.wasserstein1",
                wasserstein1(r, rdf_ref[key], rdf_model[key]),
                "A",
                status=Status.PROVISIONAL,
            )
            add(
                f"L3a.rdf.{key}.jensen_shannon",
                jensen_shannon(rdf_ref[key], rdf_model[key]),
                "-",
                status=Status.PROVISIONAL,
            )
        else:
            for field_name, unit in (("r_peak", "A"), ("g_peak", "-"), ("fwhm", "A"), ("first_minimum", "A")):
                add(f"L3a.rdf_reference.{key}.{field_name}", pr[field_name], unit, status=Status.OK)

    # -- 2. angular distribution ------------------------------------------
    adf_ref = bond_angle_distribution(reference, params, stride=stride)
    adf_model = bond_angle_distribution(model_traj, params, stride=model_stride) if model_traj else None
    if adf_model is not None:
        for species in [s for s in adf_ref if s not in ("angle",)]:
            add(
                f"L3a.adf.{species}.jensen_shannon",
                jensen_shannon(adf_ref[species], adf_model[species]),
                "-",
                status=Status.PROVISIONAL,
            )
            add(
                f"L3a.adf.{species}.wasserstein1",
                wasserstein1(adf_ref["angle"], adf_ref[species], adf_model[species]),
                "deg",
                status=Status.PROVISIONAL,
            )

    # -- 3. coordination numbers ------------------------------------------
    cn_ref = coordination_distribution(reference, params, stride=stride)
    cn_model = coordination_distribution(model_traj, params, stride=model_stride) if model_traj else None
    cn_values = cn_ref["cn_values"]
    for species in [s for s in cn_ref if s not in ("cn_values", "_meta")]:
        mean_ref = float(np.sum(cn_values * cn_ref[species]))
        if cn_model is not None:
            mean_model = float(np.sum(cn_values * cn_model[species]))
            add(
                f"L3a.cn.{species}.mean",
                mean_model,
                "-",
                status=Status.PROVISIONAL,
                extra={"reference": mean_ref, "model": mean_model, "difference": mean_model - mean_ref},
            )
            add(
                f"L3a.cn.{species}.jensen_shannon",
                jensen_shannon(cn_ref[species], cn_model[species]),
                "-",
                status=Status.PROVISIONAL,
                extra={"note": "distance between the full CN distributions, not just the means"},
            )
        else:
            add(f"L3a.cn_reference.{species}.mean", mean_ref, "-", status=Status.OK)

    # -- 4. homopolar bonds ------------------------------------------------
    homo_ref = homopolar_fraction(reference, params, stride=stride)
    homo_model = homopolar_fraction(model_traj, params, stride=model_stride) if model_traj else None
    for key, value in homo_ref.items():
        if homo_model is not None:
            add(
                f"L3a.homopolar.{key}",
                homo_model[key],
                "fraction of bonds",
                status=Status.PROVISIONAL,
                extra={"reference": value, "model": homo_model[key], "difference": homo_model[key] - value},
            )
        else:
            add(f"L3a.homopolar_reference.{key}", value, "fraction of bonds", status=Status.OK)

    # -- 5. tetrahedral order ---------------------------------------------
    tetra_ref = tetrahedral_order(reference, params, species=tetrahedral_species, stride=stride)
    tetra_model = (
        tetrahedral_order(model_traj, params, species=tetrahedral_species, stride=model_stride)
        if model_traj
        else None
    )
    if tetra_model is not None and tetra_ref.size and tetra_model.size:
        distances = sample_distances(tetra_ref, tetra_model)
        add(
            f"L3a.tetrahedral_q.{tetrahedral_species}.mean",
            float(tetra_model.mean()),
            "-",
            status=Status.PROVISIONAL,
            extra={
                "reference": float(tetra_ref.mean()),
                "model": float(tetra_model.mean()),
                "reference_fraction_q_gt_0.8": float((tetra_ref > 0.8).mean()),
                "model_fraction_q_gt_0.8": float((tetra_model > 0.8).mean()),
            },
        )
        add(
            f"L3a.tetrahedral_q.{tetrahedral_species}.wasserstein1",
            distances["wasserstein1"],
            "-",
            status=Status.PROVISIONAL,
        )
    elif tetra_ref.size:
        add(
            f"L3a.tetrahedral_q_reference.{tetrahedral_species}.mean",
            float(tetra_ref.mean()),
            "-",
            status=Status.OK,
        )

    if n_independent_references < 2:
        for item in results:
            if item.status is Status.PROVISIONAL:
                item.extra.setdefault(
                    "grading_note",
                    "provisional: one reference trajectory only, sigma_ref unknown -- no formal grade",
                )

    return L3aReport(
        params=params,
        rdf_ref=rdf_ref,
        rdf_model=rdf_model,
        adf_ref=adf_ref,
        adf_model=adf_model,
        cn_ref=cn_ref,
        cn_model=cn_model,
        homopolar_ref=homo_ref,
        homopolar_model=homo_model,
        tetra_ref=tetra_ref,
        tetra_model=tetra_model,
        peaks_ref=peaks_ref,
        peaks_model=peaks_model,
        results=results,
    )
