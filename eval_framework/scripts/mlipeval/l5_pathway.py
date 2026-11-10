"""L5 -- process and phase-transition pathway comparison.

Scope (``validation_methods_L0-L5.md`` L5), phase-1 version:

1. run the MLIP under the *same* temperature programme as the AIMD reference;
2. compare RDF peaks, coordination number, four-ring count, tetrahedral Ge and
   volume/energy as a function of time (not only the end state);
3. compare the final amorphous structure -- no crystallisation rate, no
   nucleation rate.

Because only one melt-quench trajectory exists, every metric here is emitted as
``provisional``: with a single reference there is no ``sigma_ref`` and therefore
no formal grade.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .io_vasp import Trajectory
from .l3b_medium_range import king_rings
from .neighbors import (
    StructureParams,
    coordination_numbers,
    first_peak,
    partial_rdf,
    tetrahedral_order,
)
from .schema import EvidenceTier, MetricResult, ResultSet, Status


def _result(metric_id: str, value, unit: str, **kw) -> MetricResult:
    kw.setdefault("evidence_tier", EvidenceTier.A)
    kw.setdefault("status", Status.PROVISIONAL)
    return MetricResult(metric_id=metric_id, level="L5", value=value, unit=unit, **kw)


# --------------------------------------------------------------------------
# time-resolved observables
# --------------------------------------------------------------------------
@dataclass
class PathwayTrace:
    """Structural observables as a function of trajectory progress.

    Each sample is averaged over ``average_window`` consecutive frames, and the
    scatter within that window is kept in :attr:`std`.  That scatter is the
    reference's own noise level, and it is what the divergence test compares
    against -- a fixed tolerance cannot work here, because the single-frame
    scatter of e.g. the Ge-Te RDF peak position (~0.25 A) is several times
    larger than any physically interesting shift.
    """

    name: str
    frame_index: np.ndarray
    progress: np.ndarray                 # 0..1 along the programme
    time_fs: np.ndarray
    nominal_temperature: np.ndarray
    values: dict[str, np.ndarray]
    std: dict[str, np.ndarray]
    average_window: int

    def as_dict(self) -> dict[str, np.ndarray]:
        out = {
            "progress": self.progress,
            "time_fs": self.time_fs,
            "nominal_temperature_K": self.nominal_temperature,
        }
        out.update(self.values)
        return out

    def sigma(self, key: str) -> np.ndarray:
        return self.std[key]


#: Observables tracked along the programme.
PATHWAY_OBSERVABLES = (
    "mean_cn_all",
    "mean_cn_Ge",
    "mean_cn_Sb",
    "mean_cn_Te",
    "four_rings_per_atom",
    "tetrahedral_fraction",
    "rdf_first_peak_A",
    "volume_per_atom_A3",
    "potential_energy_per_atom_eV",
)


def temperature_programme(
    n_points: int,
    segments: Sequence[tuple[float, float, float]],
) -> np.ndarray:
    """Nominal temperature at ``n_points`` equally spaced progress values.

    ``segments`` is ``[(duration_fs, T_start_K, T_end_K), ...]``; the melt-quench
    protocol is expressed directly in these terms so the AIMD and MLIP sides can
    be plotted against the same axis.
    """
    durations = np.array([s[0] for s in segments], dtype=float)
    total = durations.sum()
    edges = np.concatenate([[0.0], np.cumsum(durations)]) / total
    progress = np.linspace(0.0, 1.0, n_points)
    out = np.empty(n_points)
    for k, p in enumerate(progress):
        idx = int(np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(segments) - 1))
        span = edges[idx + 1] - edges[idx]
        local = (p - edges[idx]) / span if span > 0 else 0.0
        t0, t1 = segments[idx][1], segments[idx][2]
        out[k] = t0 + local * (t1 - t0)
    return out


def _frame_observables(
    traj: Trajectory,
    params: StructureParams,
    index: int,
    *,
    tetrahedral_species: str,
    tetrahedral_threshold: float,
) -> dict[str, float]:
    """Per-frame structural observables (RDF peak is handled separately)."""
    atoms = traj.atoms(int(index))
    symbols = np.asarray(traj.symbols)
    cn = coordination_numbers(atoms, params)
    out = {"mean_cn_all": float(cn.mean())}
    for species in traj.species:
        out[f"mean_cn_{species}"] = float(cn[symbols == species].mean())

    rings = king_rings(atoms, params)
    out["four_rings_per_atom"] = len(rings.get(4, [])) / len(atoms)

    q = tetrahedral_order(traj.slice(int(index), int(index) + 1), params, species=tetrahedral_species)
    out["tetrahedral_fraction"] = float((q > tetrahedral_threshold).mean()) if q.size else np.nan

    out["volume_per_atom_A3"] = atoms.get_volume() / len(atoms)
    out["potential_energy_per_atom_eV"] = (
        float(traj.energies[int(index)] / len(atoms)) if traj.energies is not None else np.nan
    )
    return out


def trace_pathway(
    traj: Trajectory,
    params: StructureParams,
    *,
    n_samples: int = 40,
    average_window: int = 8,
    tetrahedral_species: str = "Ge",
    tetrahedral_threshold: float = 0.8,
    reference_pair: str = "Ge-Te",
    segments: Sequence[tuple[float, float, float]] | None = None,
    name: str | None = None,
    cache: bool = True,
) -> PathwayTrace:
    """Sample structural observables at ``n_samples`` points along a trajectory.

    Every observable is averaged over ``average_window`` consecutive frames
    centred on the sample, and the within-window standard deviation is retained.
    Without that averaging the trace is dominated by single-frame noise: the
    Ge-Te RDF first peak alone scatters by ~0.25 A frame to frame.
    """
    if cache:
        from .memo import cached_arrays, trajectory_identity

        stored = cached_arrays(
            "trace_pathway",
            {"traj": trajectory_identity(traj), "params": params.to_dict(),
             "n_samples": n_samples, "average_window": average_window,
             "species": tetrahedral_species, "threshold": tetrahedral_threshold,
             "pair": reference_pair,
             "segments": [list(s) for s in segments] if segments else None},
            lambda: _trace_to_mapping(
                trace_pathway(traj, params, n_samples=n_samples, average_window=average_window,
                              tetrahedral_species=tetrahedral_species,
                              tetrahedral_threshold=tetrahedral_threshold,
                              reference_pair=reference_pair, segments=segments,
                              name=name, cache=False)
            ),
        )
        return _trace_from_mapping(stored)

    centres = np.unique(np.linspace(0, len(traj) - 1, n_samples).astype(int))
    dt = (traj.timestep_fs or 1.0) * traj.sample_stride
    half = max(1, average_window // 2)

    values: dict[str, list[float]] = {k: [] for k in PATHWAY_OBSERVABLES}
    stds: dict[str, list[float]] = {k: [] for k in PATHWAY_OBSERVABLES}

    for centre in centres:
        lo = max(0, int(centre) - half)
        hi = min(len(traj), lo + average_window)
        lo = max(0, hi - average_window)
        frames = range(lo, hi)

        samples: dict[str, list[float]] = {k: [] for k in PATHWAY_OBSERVABLES if k != "rdf_first_peak_A"}
        for index in frames:
            for key, value in _frame_observables(
                traj, params, index,
                tetrahedral_species=tetrahedral_species,
                tetrahedral_threshold=tetrahedral_threshold,
            ).items():
                samples.setdefault(key, []).append(value)

        # The RDF peak needs the whole window at once, so its scatter is
        # estimated by splitting the window in halves.
        window_traj = traj.slice(lo, hi)
        rdf = partial_rdf(window_traj, params)
        peak = first_peak(rdf["r"], rdf[reference_pair])["r_peak"]
        halves = []
        mid = lo + (hi - lo) // 2
        for a, b in ((lo, mid), (mid, hi)):
            if b - a >= 2:
                half_rdf = partial_rdf(traj.slice(a, b), params)
                halves.append(first_peak(half_rdf["r"], half_rdf[reference_pair])["r_peak"])
        samples["rdf_first_peak_A"] = halves if halves else [peak]

        for key in PATHWAY_OBSERVABLES:
            data = np.asarray(samples.get(key, [np.nan]), dtype=float)
            data = data[np.isfinite(data)]
            if key == "rdf_first_peak_A":
                values[key].append(float(peak))
                stds[key].append(float(data.std(ddof=1)) if data.size > 1 else np.nan)
            else:
                values[key].append(float(data.mean()) if data.size else np.nan)
                stds[key].append(float(data.std(ddof=1)) if data.size > 1 else np.nan)

    nominal = (
        temperature_programme(len(centres), segments)
        if segments
        else (traj.temperatures[centres] if traj.temperatures is not None else np.full(len(centres), np.nan))
    )

    return PathwayTrace(
        name=name or traj.name,
        frame_index=centres,
        progress=centres / max(len(traj) - 1, 1),
        time_fs=centres * dt,
        nominal_temperature=np.asarray(nominal, dtype=float),
        values={k: np.asarray(v) for k, v in values.items()},
        std={k: np.asarray(v) for k, v in stds.items()},
        average_window=average_window,
    )


def _trace_to_mapping(trace: PathwayTrace) -> dict:
    out = {
        "name": trace.name, "frame_index": trace.frame_index, "progress": trace.progress,
        "time_fs": trace.time_fs, "nominal_temperature": trace.nominal_temperature,
        "average_window": trace.average_window, "keys": list(trace.values),
    }
    out.update({f"v_{k}": v for k, v in trace.values.items()})
    out.update({f"s_{k}": v for k, v in trace.std.items()})
    return out


def _trace_from_mapping(stored: dict) -> PathwayTrace:
    keys = list(stored["keys"])
    return PathwayTrace(
        name=str(stored["name"]),
        frame_index=np.asarray(stored["frame_index"]),
        progress=np.asarray(stored["progress"]),
        time_fs=np.asarray(stored["time_fs"]),
        nominal_temperature=np.asarray(stored["nominal_temperature"]),
        values={k: np.asarray(stored[f"v_{k}"]) for k in keys},
        std={k: np.asarray(stored[f"s_{k}"]) for k in keys},
        average_window=int(stored["average_window"]),
    )


# --------------------------------------------------------------------------
# divergence
# --------------------------------------------------------------------------
def divergence_band(
    reference_std: np.ndarray,
    *,
    sigma_factor: float = 3.0,
    floor: float = 0.0,
) -> np.ndarray:
    """Tolerance band from the reference's own fluctuation, not a magic number.

    ``sigma_factor`` multiplies the within-window scatter of the reference
    trace.  A model that stays inside this band is indistinguishable from the
    reference given the length of the trajectories actually available.
    """
    sigma = np.asarray(reference_std, dtype=float)
    sigma = np.where(np.isfinite(sigma), sigma, np.nanmedian(sigma[np.isfinite(sigma)]) if np.isfinite(sigma).any() else 0.0)
    return np.maximum(sigma_factor * sigma, floor)


def first_divergence(
    progress: np.ndarray,
    reference: np.ndarray,
    model: np.ndarray,
    *,
    tolerance: float | np.ndarray,
    consecutive: int = 2,
) -> dict[str, float | None]:
    """Where along the programme the model first leaves the reference band.

    ``tolerance`` may be a scalar or a per-sample array (the usual case: the
    band derived from the reference's own scatter).  ``consecutive`` samples
    must exceed it, so a single noisy point does not count as a divergence.
    The returned progress value is what the failure-export stage uses to pick
    continuous frames *around* the departure, rather than only the final
    anomalous structure.
    """
    delta = np.abs(np.asarray(model, dtype=float) - np.asarray(reference, dtype=float))
    band = np.broadcast_to(np.asarray(tolerance, dtype=float), delta.shape)
    with np.errstate(invalid="ignore", divide="ignore"):
        normalised = np.where(band > 0, delta / band, np.inf)
    over = np.isfinite(delta) & (delta > band)
    run = 0
    for k, flag in enumerate(over):
        run = run + 1 if flag else 0
        if run >= consecutive:
            start = k - consecutive + 1
            return {
                "progress": float(progress[start]),
                "index": int(start),
                "delta_at_divergence": float(delta[start]),
                "band_at_divergence": float(band[start]),
                "max_delta": float(np.nanmax(delta)),
                "max_normalised": float(np.nanmax(normalised[np.isfinite(normalised)])),
            }
    return {
        "progress": None,
        "index": None,
        "delta_at_divergence": None,
        "band_at_divergence": None,
        "max_delta": float(np.nanmax(delta)) if delta.size else None,
        "max_normalised": float(np.nanmax(normalised[np.isfinite(normalised)])) if delta.size else None,
    }


#: Floors on the tolerance band, so a fluke of small scatter cannot make the
#: band collapse to zero.  The band itself is ``sigma_factor * sigma_ref``.
PATHWAY_TOLERANCE_FLOORS = {
    "mean_cn_all": 0.05,
    "mean_cn_Ge": 0.08,
    "mean_cn_Sb": 0.08,
    "mean_cn_Te": 0.08,
    "four_rings_per_atom": 0.02,
    "tetrahedral_fraction": 0.02,
    "rdf_first_peak_A": 0.03,
    "volume_per_atom_A3": 0.01,
}


# --------------------------------------------------------------------------
# battery
# --------------------------------------------------------------------------
@dataclass
class L5Report:
    trace_ref: PathwayTrace
    trace_model: PathwayTrace | None
    bands: dict[str, np.ndarray]
    divergences: dict[str, dict[str, float | None]]
    final_state_metrics: list[MetricResult]
    results: ResultSet


def compare_pathway(
    reference: Trajectory,
    params: StructureParams,
    *,
    model_traj: Trajectory | None = None,
    n_samples: int = 40,
    average_window: int = 8,
    sigma_factor: float = 3.0,
    segments: Sequence[tuple[float, float, float]] | None = None,
    state_label: str = "melt_quench",
    observables: Sequence[str] | None = None,
    n_independent_references: int = 1,
) -> L5Report:
    """Compare a full temperature programme, not only its end state.

    The tolerance band for each observable is ``sigma_factor`` times the
    reference's own within-window scatter (floored by
    :data:`PATHWAY_TOLERANCE_FLOORS`).  A fixed tolerance would not work: the
    frame-to-frame scatter of the reference is itself several times larger than
    the differences being looked for.
    """
    observables = list(observables or PATHWAY_TOLERANCE_FLOORS)
    results = ResultSet(
        "L5",
        state=state_label,
        frozen_params=params.to_dict(),
        reference=reference.describe(),
        model_trajectory=model_traj.describe() if model_traj else None,
        temperature_segments=[list(s) for s in segments] if segments else None,
        n_independent_references=n_independent_references,
        average_window_frames=average_window,
        sigma_factor=sigma_factor,
        band_definition="tolerance = max(sigma_factor * within-window std of the reference, floor)",
        note=(
            "single melt-quench chain on the reference side: every pathway metric is provisional "
            "until independent replicas exist (validation_methods_L0-L5.md L5)"
        ),
    )
    tier = EvidenceTier.A if model_traj is not None else EvidenceTier.S
    ref_pid = reference.protocol.protocol_id if reference.protocol else None
    mod_pid = model_traj.protocol.protocol_id if (model_traj and model_traj.protocol) else None

    def add(metric_id, value, unit, **kw):
        kw.setdefault("evidence_tier", tier)
        kw.setdefault("reference_protocol", ref_pid)
        kw.setdefault("model_protocol", mod_pid)
        kw.setdefault("status", Status.PROVISIONAL)
        return results.add(
            MetricResult(metric_id=f"{metric_id}@{state_label}", level="L5", value=value, unit=unit, **kw)
        )

    trace_ref = trace_pathway(reference, params, n_samples=n_samples, average_window=average_window,
                              segments=segments, name="AIMD")
    trace_model = (
        trace_pathway(model_traj, params, n_samples=n_samples, average_window=average_window,
                      segments=segments, name="MLIP")
        if model_traj
        else None
    )

    bands: dict[str, np.ndarray] = {}
    divergences: dict[str, dict[str, float | None]] = {}
    for key in observables:
        if key not in trace_ref.values:
            continue
        band = divergence_band(trace_ref.std[key], sigma_factor=sigma_factor,
                               floor=PATHWAY_TOLERANCE_FLOORS.get(key, 0.0))
        bands[key] = band
        if trace_model is None:
            add(f"L5.pathway_reference.{key}.final", float(trace_ref.values[key][-1]), "-", status=Status.OK)
            continue

        div = first_divergence(trace_ref.progress, trace_ref.values[key], trace_model.values[key],
                               tolerance=band)
        divergences[key] = div
        add(
            f"L5.pathway.{key}.first_divergence_progress",
            div["progress"],
            "fraction of programme",
            status=Status.PROVISIONAL if div["progress"] is None else Status.FAILED,
            failure_reason=(
                None
                if div["progress"] is None
                else (
                    f"model leaves the {sigma_factor:g}-sigma reference band at "
                    f"{div['progress']:.2f} of the programme "
                    f"(delta {div['delta_at_divergence']:.4g} vs band {div['band_at_divergence']:.4g})"
                )
            ),
            extra={
                "band_mean": float(np.nanmean(band)),
                "reference_sigma_mean": float(np.nanmean(trace_ref.std[key])),
                "max_delta": div["max_delta"],
                "max_delta_in_sigma_units": div["max_normalised"],
                "note": "export continuous frames around this point, not only the final structure",
            },
        )
        add(
            f"L5.pathway.{key}.final_state_error",
            float(trace_model.values[key][-1] - trace_ref.values[key][-1]),
            "-",
            extra={
                "reference_final": float(trace_ref.values[key][-1]),
                "model_final": float(trace_model.values[key][-1]),
                "reference_sigma_final": float(trace_ref.std[key][-1]),
                "within_reference_scatter": bool(
                    abs(trace_model.values[key][-1] - trace_ref.values[key][-1]) <= band[-1]
                ),
            },
        )

    final_metrics = [r for r in results if r.metric_id.endswith(f"final_state_error@{state_label}")]

    if n_independent_references < 2:
        for item in results:
            if item.status is Status.PROVISIONAL:
                item.extra.setdefault(
                    "grading_note",
                    "provisional: single melt-quench reference; independent replicas required for a grade",
                )

    return L5Report(
        trace_ref=trace_ref,
        trace_model=trace_model,
        bands=bands,
        divergences=divergences,
        final_state_metrics=final_metrics,
        results=results,
    )
