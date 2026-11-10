"""L4 -- dynamics and transport at one fixed-temperature state point.

Scope (``validation_methods_L0-L5.md`` L4):

1. species-resolved MSD;
2. self-diffusion coefficients;
3. the fit window and the log-log slope inside that window;
4. ``log10(D)`` difference between AIMD and MLIP.

Guard rail
----------
:func:`assert_equilibrium_segment` refuses a heating/quenching trajectory as a
diffusion reference.  The framework is explicit that the existing GST225
ramp trajectories cannot be used for equilibrium diffusion, so the check is part
of the module rather than a note in a notebook.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io_vasp import Trajectory
from .schema import EvidenceTier, MetricResult, ResultSet, Status


def _result(metric_id: str, value, unit: str, **kw) -> MetricResult:
    kw.setdefault("evidence_tier", EvidenceTier.A)
    kw.setdefault("status", Status.PROVISIONAL)
    return MetricResult(metric_id=metric_id, level="L4", value=value, unit=unit, **kw)


# --------------------------------------------------------------------------
# unwrapping
# --------------------------------------------------------------------------
def unwrap(traj: Trajectory) -> np.ndarray:
    """Continuous Cartesian trajectory from wrapped coordinates.

    XDATCAR stores wrapped fractional coordinates; the minimum-image difference
    between consecutive frames is accumulated.  This is valid as long as no atom
    moves more than half a cell edge between stored frames -- which
    :func:`max_displacement_per_frame` lets the caller verify.
    """
    positions = traj.positions
    cells = traj.cells
    n_frames, n_atoms, _ = positions.shape
    out = np.empty_like(positions)
    out[0] = positions[0]
    for k in range(1, n_frames):
        cell = cells[k]
        inverse = np.linalg.inv(cell)
        delta = (positions[k] - positions[k - 1]) @ inverse
        delta -= np.round(delta)
        out[k] = out[k - 1] + delta @ cell
    return out


def max_displacement_per_frame(traj: Trajectory) -> float:
    """Largest single-frame displacement in units of half the shortest cell edge.

    A value >= 1 means unwrapping is ambiguous and the sampling stride is too
    coarse for a reliable MSD.
    """
    positions, cells = traj.positions, traj.cells
    if len(traj) < 2:
        return 0.0
    inverse = np.linalg.inv(cells[1])
    delta = (positions[1:] - positions[:-1]) @ inverse
    delta -= np.round(delta)
    cartesian = np.einsum("fai,ij->faj", delta, cells[1])
    half_min_edge = 0.5 * float(np.min(np.linalg.norm(cells[1], axis=1)))
    return float(np.linalg.norm(cartesian, axis=-1).max() / half_min_edge)


# --------------------------------------------------------------------------
# MSD
# --------------------------------------------------------------------------
@dataclass
class MSDResult:
    time_fs: np.ndarray
    msd: dict[str, np.ndarray]  # A^2, per species plus "all"
    n_origins: np.ndarray
    label: str = ""

    def species(self) -> list[str]:
        return [s for s in self.msd if s != "all"]


def mean_squared_displacement(
    traj: Trajectory,
    *,
    max_lag_fraction: float = 0.5,
    use_multiple_origins: bool = True,
    cache: bool = True,
) -> MSDResult:
    """Species-resolved MSD, averaged over time origins.

    Multiple time origins reduce the variance at long lag times; the number of
    origins per lag is returned so a notebook can show where the statistics run
    out instead of fitting into the noisy tail.
    """
    if cache:
        from .memo import cached_arrays, trajectory_identity

        stored = cached_arrays(
            "mean_squared_displacement",
            {"traj": trajectory_identity(traj), "max_lag_fraction": max_lag_fraction,
             "multiple_origins": use_multiple_origins},
            lambda: _msd_to_mapping(
                mean_squared_displacement(traj, max_lag_fraction=max_lag_fraction,
                                          use_multiple_origins=use_multiple_origins, cache=False)
            ),
        )
        return _msd_from_mapping(stored)

    unwrapped = unwrap(traj)
    n_frames = unwrapped.shape[0]
    max_lag = max(1, int(n_frames * max_lag_fraction))
    symbols = np.asarray(traj.symbols)
    dt = (traj.timestep_fs or 1.0) * traj.sample_stride

    msd: dict[str, np.ndarray] = {}
    origins = np.zeros(max_lag)
    groups = {s: np.flatnonzero(symbols == s) for s in traj.species}
    groups["all"] = np.arange(len(symbols))
    for name in groups:
        msd[name] = np.zeros(max_lag)

    for lag in range(max_lag):
        if use_multiple_origins:
            delta = unwrapped[lag:] - unwrapped[: n_frames - lag]
            origins[lag] = n_frames - lag
        else:
            delta = (unwrapped[lag] - unwrapped[0])[None]
            origins[lag] = 1
        squared = np.sum(delta**2, axis=-1)  # (n_origins, n_atoms)
        for name, idx in groups.items():
            msd[name][lag] = float(squared[:, idx].mean())

    return MSDResult(time_fs=np.arange(max_lag) * dt, msd=msd, n_origins=origins, label=traj.name)


def _msd_to_mapping(result: MSDResult) -> dict:
    out = {"time_fs": result.time_fs, "n_origins": result.n_origins, "label": result.label,
           "species": list(result.msd)}
    out.update({f"msd_{k}": v for k, v in result.msd.items()})
    return out


def _msd_from_mapping(stored: dict) -> MSDResult:
    species = list(stored["species"])
    return MSDResult(
        time_fs=np.asarray(stored["time_fs"]),
        msd={s: np.asarray(stored[f"msd_{s}"]) for s in species},
        n_origins=np.asarray(stored["n_origins"]),
        label=str(stored.get("label", "")),
    )


# --------------------------------------------------------------------------
# diffusion coefficient
# --------------------------------------------------------------------------
@dataclass
class DiffusionFit:
    species: str
    d_cm2_per_s: float
    d_A2_per_fs: float
    fit_window_fs: tuple[float, float]
    loglog_slope: float
    r2: float
    n_points: int
    diffusive: bool                 # slope within tolerance of 1
    marginal: bool = False          # slope outside tolerance but within 2x it
    d_window_spread: float = float("nan")  # relative spread of D across sub-windows

    @property
    def regime(self) -> str:
        if self.diffusive:
            return "diffusive"
        return "marginal" if self.marginal else "not diffusive"

    def to_dict(self) -> dict[str, object]:
        return {
            "species": self.species,
            "D_cm2_per_s": self.d_cm2_per_s,
            "D_A2_per_fs": self.d_A2_per_fs,
            "fit_window_fs": list(self.fit_window_fs),
            "loglog_slope": self.loglog_slope,
            "fit_r2": self.r2,
            "n_points": self.n_points,
            "regime": self.regime,
            "diffusive_regime": self.diffusive,
            "D_relative_spread_across_subwindows": self.d_window_spread,
        }


def fit_diffusion(
    msd_result: MSDResult,
    species: str,
    *,
    window_fraction: tuple[float, float] = (0.2, 0.8),
    window_fs: tuple[float, float] | None = None,
    slope_tolerance: float = 0.15,
) -> DiffusionFit:
    """``D = MSD / (6 t)`` from a straight-line fit over an explicit window.

    The log-log slope over the same window is reported: only a slope near 1
    means the window is actually in the diffusive regime.  The verdict is
    three-way rather than binary -- ``diffusive`` within ``slope_tolerance`` of
    1, ``marginal`` out to twice that, ``not diffusive`` beyond.  A short
    trajectory routinely lands in ``marginal`` while still giving a converged
    ``D``, so ``d_window_spread`` (how much ``D`` moves across sub-windows of
    the same fit) is reported alongside: that, not the slope alone, is the
    evidence that the number has converged.

    ``window_fs`` pins the window in absolute time.  This matters whenever the
    reference and the model trajectory have different lengths: a *fractional*
    window would then fit the two sides over different physical time ranges,
    and the resulting ``D`` values would not be comparable.
    """
    t = msd_result.time_fs
    y = msd_result.msd[species]
    if window_fs is not None:
        mask = (t >= window_fs[0]) & (t <= window_fs[1])
        tw, yw = t[mask], y[mask]
    else:
        lo = int(len(t) * window_fraction[0])
        hi = max(lo + 3, int(len(t) * window_fraction[1]))
        hi = min(hi, len(t))
        tw, yw = t[lo:hi], y[lo:hi]
    ok = np.isfinite(tw) & np.isfinite(yw) & (tw > 0) & (yw > 0)
    tw, yw = tw[ok], yw[ok]
    if tw.size < 3:
        return DiffusionFit(species, np.nan, np.nan, (np.nan, np.nan), np.nan, np.nan, int(tw.size), False)

    slope, intercept = np.polyfit(tw, yw, 1)
    predicted = slope * tw + intercept
    ss_res = float(np.sum((yw - predicted) ** 2))
    ss_tot = float(np.sum((yw - yw.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    log_slope = float(np.polyfit(np.log(tw), np.log(yw), 1)[0])
    d_a2_per_fs = float(slope / 6.0)
    # 1 A^2/fs = 1e-16 cm^2 / 1e-15 s = 0.1 cm^2/s
    d_cm2_per_s = d_a2_per_fs * 0.1

    # How much does D move if the window is shifted?  A slope slightly above 1
    # means MSD still has curvature, but if D is stable across sub-windows the
    # number is nonetheless converged -- and that is the property worth quoting.
    thirds = np.array_split(np.arange(tw.size), 3)
    sub = [
        float(np.polyfit(tw[idx], yw[idx], 1)[0] / 6.0) for idx in thirds if idx.size >= 3
    ]
    spread = float((max(sub) - min(sub)) / abs(d_a2_per_fs)) if len(sub) > 1 and d_a2_per_fs else float("nan")

    deviation = abs(log_slope - 1.0)
    return DiffusionFit(
        species=species,
        d_cm2_per_s=d_cm2_per_s,
        d_A2_per_fs=d_a2_per_fs,
        fit_window_fs=(float(tw[0]), float(tw[-1])),
        loglog_slope=log_slope,
        r2=float(r2),
        n_points=int(tw.size),
        diffusive=bool(deviation <= slope_tolerance),
        marginal=bool(slope_tolerance < deviation <= 2.0 * slope_tolerance),
        d_window_spread=spread,
    )


# --------------------------------------------------------------------------
# equilibrium guard
# --------------------------------------------------------------------------
def assert_equilibrium_segment(
    traj: Trajectory,
    *,
    max_temperature_drift_k: float = 100.0,
) -> MetricResult:
    """Reject a ramping trajectory as an equilibrium diffusion reference.

    Uses the recorded per-frame temperature when available, otherwise falls back
    to the protocol: a protocol with ``temperature_k`` given as a ``(start, end)``
    pair is a ramp by construction.
    """
    protocol = traj.protocol
    temperature = protocol.temperature_k if protocol else None
    if isinstance(temperature, (tuple, list)) and len(temperature) == 2 and temperature[0] != temperature[1]:
        return _result(
            "L4.equilibrium_segment",
            False,
            "-",
            status=Status.FAILED,
            evidence_tier=EvidenceTier.S,
            failure_reason=(
                f"protocol is a temperature ramp {temperature[0]} -> {temperature[1]} K; "
                "a ramp cannot be used for equilibrium diffusion"
            ),
        )
    if traj.temperatures is not None and traj.temperatures.size > 4:
        temps = traj.temperatures
        half = len(temps) // 2
        drift = abs(float(temps[half:].mean() - temps[:half].mean()))
        ok = drift <= max_temperature_drift_k
        return _result(
            "L4.equilibrium_segment",
            ok,
            "-",
            status=Status.OK if ok else Status.FAILED,
            evidence_tier=EvidenceTier.S,
            failure_reason=None if ok else f"temperature drifts by {drift:.0f} K between the two halves",
            extra={
                "mean_T_K": float(temps.mean()),
                "std_T_K": float(temps.std()),
                "half_to_half_drift_K": drift,
            },
        )
    return _result(
        "L4.equilibrium_segment",
        None,
        "-",
        status=Status.SKIPPED,
        evidence_tier=EvidenceTier.S,
        failure_reason="no per-frame temperature and no protocol temperature: equilibrium unverified",
    )


# --------------------------------------------------------------------------
# battery
# --------------------------------------------------------------------------
@dataclass
class L4Report:
    msd_ref: MSDResult
    msd_model: MSDResult | None
    fits_ref: dict[str, DiffusionFit]
    fits_model: dict[str, DiffusionFit]
    results: ResultSet


def analyse_dynamics(
    reference: Trajectory,
    *,
    model_traj: Trajectory | None = None,
    window_fraction: tuple[float, float] = (0.2, 0.8),
    state_label: str = "liquid",
    n_independent_references: int = 1,
) -> L4Report:
    """MSD and self-diffusion for one fixed-temperature state point."""
    results = ResultSet(
        "L4",
        state=state_label,
        reference=reference.describe(),
        model_trajectory=model_traj.describe() if model_traj else None,
        fit_window_fraction=list(window_fraction),
        n_independent_references=n_independent_references,
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
            MetricResult(metric_id=f"{metric_id}@{state_label}", level="L4", value=value, unit=unit, **kw)
        )

    guard = assert_equilibrium_segment(reference)
    guard.metric_id = f"{guard.metric_id}@{state_label}.reference"
    results.add(guard)
    if model_traj is not None:
        guard_model = assert_equilibrium_segment(model_traj)
        guard_model.metric_id = f"{guard_model.metric_id}@{state_label}.model"
        results.add(guard_model)

    add(
        "L4.unwrap_safety",
        max_displacement_per_frame(reference),
        "half-cell-edge",
        status=Status.OK if max_displacement_per_frame(reference) < 0.5 else Status.FAILED,
        evidence_tier=EvidenceTier.S,
        failure_reason=None if max_displacement_per_frame(reference) < 0.5 else "sampling stride too coarse to unwrap",
    )

    msd_ref = mean_squared_displacement(reference)
    msd_model = mean_squared_displacement(model_traj) if model_traj else None

    # Pin the fit window in absolute time over the span both sides cover, so a
    # shorter model trajectory does not silently get fitted somewhere else.
    common_max_fs = msd_ref.time_fs[-1]
    if msd_model is not None:
        common_max_fs = min(common_max_fs, msd_model.time_fs[-1])
    window_fs = (window_fraction[0] * common_max_fs, window_fraction[1] * common_max_fs)
    results.context["fit_window_fs"] = list(window_fs)
    results.context["common_lag_span_fs"] = float(common_max_fs)
    add(
        "L4.fit_window",
        list(window_fs),
        "fs",
        status=Status.OK,
        evidence_tier=EvidenceTier.S,
        extra={
            "common_lag_span_fs": float(common_max_fs),
            "reference_max_lag_fs": float(msd_ref.time_fs[-1]),
            "model_max_lag_fs": float(msd_model.time_fs[-1]) if msd_model is not None else None,
            "note": "absolute window shared by both sides; a fractional window would fit different time ranges",
        },
    )

    fits_ref: dict[str, DiffusionFit] = {}
    fits_model: dict[str, DiffusionFit] = {}
    for species in list(msd_ref.msd):
        fit_ref = fit_diffusion(msd_ref, species, window_fs=window_fs)
        fits_ref[species] = fit_ref
        if msd_model is not None:
            fit_model = fit_diffusion(msd_model, species, window_fs=window_fs)
            fits_model[species] = fit_model
            add(
                f"L4.diffusion.{species}.D",
                fit_model.d_cm2_per_s,
                "cm^2/s",
                status=Status.FAILED if fit_model.regime == "not diffusive" else Status.PROVISIONAL,
                failure_reason=(
                    None
                    if fit_model.diffusive
                    else (
                        f"log-log slope {fit_model.loglog_slope:.2f} is {fit_model.regime} in this window; "
                        f"D moves by {100 * fit_model.d_window_spread:.0f}% across sub-windows"
                    )
                ),
                extra={"reference": fit_ref.to_dict(), "model": fit_model.to_dict()},
            )
            if fit_ref.d_cm2_per_s > 0 and fit_model.d_cm2_per_s > 0:
                delta = float(np.log10(fit_model.d_cm2_per_s) - np.log10(fit_ref.d_cm2_per_s))
                add(
                    f"L4.diffusion.{species}.delta_log10D",
                    delta,
                    "decades",
                    extra={
                        "reference_D_cm2_per_s": fit_ref.d_cm2_per_s,
                        "model_D_cm2_per_s": fit_model.d_cm2_per_s,
                        "note": "the L4 scoring quantity; requires >= 2 independent references to grade",
                    },
                )
        else:
            add(
                f"L4.diffusion_reference.{species}.D",
                fit_ref.d_cm2_per_s,
                "cm^2/s",
                status=Status.FAILED if fit_ref.regime == "not diffusive" else Status.OK,
                failure_reason=(
                    None if fit_ref.diffusive else f"reference fit window is {fit_ref.regime}"
                ),
                extra=fit_ref.to_dict(),
            )

    if n_independent_references < 2:
        for item in results:
            if item.status is Status.PROVISIONAL:
                item.extra.setdefault(
                    "grading_note",
                    "provisional: one reference trajectory only, sigma_ref unknown -- no formal grade",
                )

    return L4Report(msd_ref=msd_ref, msd_model=msd_model, fits_ref=fits_ref, fits_model=fits_model, results=results)
