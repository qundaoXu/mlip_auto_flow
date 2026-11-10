"""Thermodynamic quantities recovered from a bare trajectory.

Several reference trajectories in this workspace ship as XDATCAR only: the
216-atom AIMD stages have no OSZICAR or OUTCAR, and the 306-atom runs have no
INCAR either.  Without a temperature series the L4 equilibrium guard cannot do
its job and a quench cannot be plotted against its own programme.

The kinetic temperature can still be measured, because consecutive XDATCAR
frames are one MD step apart: central-difference velocities give

    T = sum_i m_i |v_i|^2 / (3 N k_B)

The finite-difference velocity underestimates a harmonic mode by
``sinc(w dt / 2)``; at 3 fs against ~150 fs phonon periods that is well under a
percent, and :func:`temperature_series` reports the correction factor so the
approximation is never silent.  The estimate scales as ``1/dt^2``, so it also
provides a consistency check on an *assumed* timestep: if the recovered
temperature of a run labelled "2000 K" comes out at 2000 K, the assumed
timestep is right.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase.data import atomic_masses, atomic_numbers
from ase.units import kB

def _mass_array(symbols) -> np.ndarray:
    return np.array([atomic_masses[atomic_numbers[s]] for s in symbols])


def _velocities(traj, *, timestep_fs: float) -> np.ndarray:
    """Central-difference velocities in A/fs, minimum-image corrected."""
    positions, cells = traj.positions, traj.cells
    n_frames = positions.shape[0]
    if n_frames < 3:
        raise ValueError("need at least three frames to central-difference velocities")
    inverse = np.linalg.inv(cells[0])
    delta = (positions[2:] - positions[:-2]) @ inverse
    delta -= np.round(delta)
    cartesian = delta @ cells[0]
    return cartesian / (2.0 * timestep_fs)


@dataclass
class TemperatureSeries:
    time_fs: np.ndarray
    temperature: np.ndarray
    timestep_fs: float
    mean: float
    std: float
    half_to_half_drift: float
    finite_difference_note: str

    def summary(self) -> dict[str, float | str]:
        return {
            "mean_T_K": self.mean,
            "std_T_K": self.std,
            "half_to_half_drift_K": self.half_to_half_drift,
            "assumed_timestep_fs": self.timestep_fs,
            "n_points": int(self.temperature.size),
            "note": self.finite_difference_note,
        }


def temperature_series(traj, *, timestep_fs: float | None = None) -> TemperatureSeries:
    """Kinetic temperature per frame, measured from the trajectory itself.

    ``timestep_fs`` defaults to the trajectory's own value.  Because the
    trajectory may have been sampled with a stride, the *effective* spacing
    ``timestep_fs * sample_stride`` is what enters the velocity estimate.
    """
    dt = (timestep_fs if timestep_fs is not None else traj.timestep_fs) or 1.0
    spacing = dt * traj.sample_stride
    velocities = _velocities(traj, timestep_fs=spacing)  # A/fs
    masses = _mass_array(traj.symbols)  # amu

    # 1 amu * (A/fs)^2 = 1.0364e-4 eV^-1 ... use ASE units explicitly:
    # ASE energy unit is eV and its time unit is such that 1 fs = ase.units.fs.
    from ase.units import fs as ASE_FS

    v_ase = velocities / ASE_FS  # A per ASE time unit
    kinetic = 0.5 * np.einsum("a,fai,fai->f", masses, v_ase, v_ase)
    n_dof = 3 * len(masses)
    temperature = 2.0 * kinetic / (n_dof * kB)

    half = temperature.size // 2
    drift = (
        float(temperature[half:].mean() - temperature[:half].mean()) if temperature.size > 3 else float("nan")
    )
    time = (np.arange(temperature.size) + 1) * spacing
    return TemperatureSeries(
        time_fs=time,
        temperature=temperature,
        timestep_fs=spacing,
        mean=float(temperature.mean()),
        std=float(temperature.std()),
        half_to_half_drift=drift,
        finite_difference_note=(
            f"central-difference velocities at {spacing:g} fs spacing; a harmonic mode of "
            f"period tau is underestimated by sinc(pi*{spacing:g}/tau), i.e. <1% for tau >~ 100 fs. "
            "T scales as 1/dt^2, so a wrong assumed timestep shows up as a wrong temperature."
        ),
    )


def attach_temperature(traj, *, timestep_fs: float | None = None):
    """Fill ``traj.temperatures`` in place from the measured kinetic temperature.

    The first and last frames have no central difference, so they inherit their
    neighbours' value; this keeps the array aligned with ``traj.positions``.
    """
    series = temperature_series(traj, timestep_fs=timestep_fs)
    padded = np.empty(len(traj))
    padded[1:-1] = series.temperature
    padded[0] = series.temperature[0]
    padded[-1] = series.temperature[-1]
    traj.temperatures = padded
    return series


def infer_timestep(traj, target_temperature_k: float, *, assumed_timestep_fs: float = 3.0) -> dict[str, float]:
    """Timestep implied by a known target temperature, since ``T`` scales as ``1/dt^2``.

    Used to test whether the project-wide 3 fs assumption is consistent with a
    trajectory whose INCAR was not preserved.
    """
    series = temperature_series(traj, timestep_fs=assumed_timestep_fs)
    if series.mean <= 0:
        return {"implied_timestep_fs": float("nan")}
    factor = np.sqrt(series.mean / target_temperature_k)
    return {
        "assumed_timestep_fs": assumed_timestep_fs,
        "measured_T_at_assumed_dt_K": series.mean,
        "target_T_K": target_temperature_k,
        "implied_timestep_fs": float(assumed_timestep_fs * factor),
        "consistent_within_5pct": bool(abs(factor - 1.0) < 0.05),
    }
