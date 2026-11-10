"""MLIP molecular dynamics under the same protocol as the AIMD reference.

The point of this module is that an MLIP trajectory and its AIMD counterpart
must differ *only* in the engine: same cell, same composition, same timestep,
same ensemble, same thermostat time constant, same production window.  Anything
that changes a number is captured in :class:`MDSettings` and echoed into the
resulting :class:`~mlipeval.io_vasp.Trajectory`'s protocol.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, fields
from typing import Callable

import numpy as np
from ase import Atoms, units

from .calculators import Model
from .io_vasp import Trajectory
from .schema import Protocol

#: VASP ``SMASS = 0`` selects a Nose period of 40 ionic steps.
VASP_SMASS0_PERIOD_STEPS = 40


@dataclass
class MDSettings:
    """Everything that changes an MD number, in one object."""

    ensemble: str = "nvt"  # "nvt" | "nve"
    temperature_k: float = 1200.0
    temperature_end_k: float | None = None  # linear ramp target (melt-quench)
    timestep_fs: float = 3.0
    n_steps: int = 1000
    sample_every: int = 5
    thermostat: str = "nose-hoover-chain"  # "nose-hoover-chain" | "langevin" | "none"
    tdamp_fs: float | None = None  # default: 40 * timestep, matching SMASS=0
    langevin_friction_per_fs: float = 0.01
    seed: int = 0
    initialise_velocities: bool = True
    remove_com_motion: bool = True
    equilibration_fraction: float = 0.5
    min_distance_floor: float = 1.0  # below this the run is declared unphysical
    #: 产生这条轨迹的 MD 引擎，"lammps" 或 "ase"。两个都是受支持的选项且物理等价，
    #: 所以必须逐条轨迹记下来：跨模型比较混了引擎就不再是"唯一变量是模型"。
    engine: str = "lammps"
    protocol_id: str = "mlip_md"
    description: str = ""

    @property
    def effective_tdamp_fs(self) -> float:
        return self.tdamp_fs if self.tdamp_fs is not None else VASP_SMASS0_PERIOD_STEPS * self.timestep_fs

    @property
    def total_time_fs(self) -> float:
        return self.n_steps * self.timestep_fs

    def to_protocol(self, model: Model) -> Protocol:
        return Protocol(
            protocol_id=self.protocol_id,
            kind="mlip_md",
            description=self.description or f"{model.spec.name} {self.ensemble.upper()} MD",
            code=model.spec.name,
            code_version=model.spec.software.get("mace", "unknown"),
            functional=model.spec.native_protocol,
            ensemble=self.ensemble.upper(),
            thermostat=(
                "none"
                if self.ensemble == "nve"
                else f"{self.thermostat} (tdamp={self.effective_tdamp_fs:g} fs)"
            ),
            timestep_fs=self.timestep_fs,
            n_steps=self.n_steps,
            temperature_k=(
                (self.temperature_k, self.temperature_end_k)
                if self.temperature_end_k is not None
                else self.temperature_k
            ),
            seed=self.seed,
            postprocessing={
                "sample_every": self.sample_every,
                "equilibration_fraction": self.equilibration_fraction,
                "md_engine": self.engine,
            },
            provenance={"checkpoint_sha256": model.spec.checkpoint_sha256, "dtype": model.spec.dtype},
        )


@dataclass
class MDResult:
    """Trajectory plus the per-step diagnostics that decide L1 pass/fail."""

    trajectory: Trajectory
    settings: MDSettings
    step: np.ndarray
    time_fs: np.ndarray
    potential_energy: np.ndarray
    kinetic_energy: np.ndarray
    total_energy: np.ndarray  # conserved quantity (incl. thermostat energy for NVT)
    temperature: np.ndarray
    min_distance: np.ndarray
    com_drift: np.ndarray
    wall_time_s: float
    crashed_at: int | None = None
    crash_reason: str | None = None

    @property
    def completed(self) -> bool:
        return self.crashed_at is None

    @property
    def n_atoms(self) -> int:
        return self.trajectory.n_atoms

    def drift_mev_per_atom_per_ps(self) -> float:
        """Least-squares slope of the conserved quantity."""
        t_ps = self.time_fs / 1000.0
        e = self.total_energy / self.n_atoms * 1000.0
        ok = np.isfinite(e) & np.isfinite(t_ps)
        if ok.sum() < 3:
            return float("nan")
        return float(np.polyfit(t_ps[ok], e[ok], 1)[0])

    def production(self) -> Trajectory:
        return self.trajectory.last_fraction(1.0 - self.settings.equilibration_fraction)

    def summary(self) -> dict[str, object]:
        return {
            "protocol_id": self.settings.protocol_id,
            "ensemble": self.settings.ensemble,
            "completed": self.completed,
            "crashed_at": self.crashed_at,
            "n_steps": int(self.step[-1]) if self.step.size else 0,
            "time_ps": float(self.time_fs[-1] / 1000.0) if self.time_fs.size else 0.0,
            "mean_T_K": float(np.nanmean(self.temperature)) if self.temperature.size else float("nan"),
            "min_distance_A": float(np.nanmin(self.min_distance)) if self.min_distance.size else float("nan"),
            "drift_meV_per_atom_per_ps": self.drift_mev_per_atom_per_ps(),
            "max_com_drift_A": float(np.nanmax(self.com_drift)) if self.com_drift.size else float("nan"),
            "wall_time_s": round(self.wall_time_s, 1),
        }


def _min_distance(atoms: Atoms) -> float:
    from .neighbors import neighbour_list

    try:
        d = neighbour_list(atoms, 3.5, "d")
        if isinstance(d, tuple):
            d = d[0]
        return float(np.min(d)) if len(d) else float("inf")
    except Exception:
        return float("nan")


class _MDStop(Exception):
    """Raised from an observer to stop a run without unwinding the trajectory."""

    def __init__(self, step: int, reason: str) -> None:
        super().__init__(reason)
        self.step = step
        self.reason = reason


def _set_nhc_temperature(dyn, temperature_k: float) -> None:
    """Retarget an ASE Nose-Hoover chain, including its thermostat masses.

    ``NoseHooverChainNVT`` exposes no setter, and ``Q`` is proportional to
    ``kT``, so both have to be updated together for a temperature ramp.
    """
    thermostat = dyn._thermostat
    kT = units.kB * temperature_k
    thermostat._kT = kT
    thermostat._Q[0] = 3 * thermostat._num_atoms_global * kT * thermostat._tdamp**2
    thermostat._Q[1:] = kT * thermostat._tdamp**2


def run_md(
    atoms: Atoms,
    model: Model,
    settings: MDSettings,
    *,
    progress: Callable[[int, int], None] | None = None,
    progress_every: int = 50,
) -> MDResult:
    """Run MLIP MD, recording a trajectory and per-step health diagnostics.

    A blow-up (non-finite energy, atoms closer than
    ``settings.min_distance_floor``) stops the run and is reported through
    ``crashed_at`` rather than raised: an unstable model is a finding, not an
    error.  Observers are attached to a single ``dyn.run`` call so that no extra
    force evaluation is paid per step.
    """
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
    from ase.md.verlet import VelocityVerlet

    rng = np.random.default_rng(settings.seed)
    work = atoms.copy()
    work.set_pbc(True)
    work.calc = model.calculator

    if settings.initialise_velocities:
        MaxwellBoltzmannDistribution(work, temperature_K=settings.temperature_k, rng=rng)
        if settings.remove_com_motion:
            Stationary(work)
            ZeroRotation(work)

    dt = settings.timestep_fs * units.fs
    thermostat_energy: Callable[[], float] = lambda: 0.0
    if settings.ensemble == "nve":
        dyn = VelocityVerlet(work, timestep=dt)
        set_temperature = None
    elif settings.thermostat == "langevin":
        from ase.md.langevin import Langevin

        dyn = Langevin(
            work,
            timestep=dt,
            temperature_K=settings.temperature_k,
            friction=settings.langevin_friction_per_fs / units.fs,
            rng=rng,
        )
        set_temperature = lambda value: dyn.set_temperature(temperature_K=value)
    else:
        from ase.md.nose_hoover_chain import NoseHooverChainNVT

        dyn = NoseHooverChainNVT(
            work,
            timestep=dt,
            temperature_K=settings.temperature_k,
            tdamp=settings.effective_tdamp_fs * units.fs,
        )
        set_temperature = lambda value: _set_nhc_temperature(dyn, value)
        thermostat_energy = lambda: float(dyn._thermostat.get_thermostat_energy())

    n_samples = settings.n_steps // settings.sample_every + 1
    positions = np.empty((n_samples, len(work), 3))
    cells = np.empty((n_samples, 3, 3))
    step_l, time_l, epot_l, ekin_l, econs_l, temp_l, dmin_l, com_l = ([] for _ in range(8))
    com0 = work.get_center_of_mass()
    state = {"sample": 0}

    def ramp() -> None:
        if settings.temperature_end_k is None or set_temperature is None or settings.n_steps == 0:
            return
        frac = min(1.0, dyn.get_number_of_steps() / settings.n_steps)
        set_temperature(settings.temperature_k + frac * (settings.temperature_end_k - settings.temperature_k))

    def sample() -> None:
        step = dyn.get_number_of_steps()
        if state["sample"] >= n_samples:
            return
        try:
            epot = float(work.get_potential_energy())
        except Exception as exc:
            raise _MDStop(step, f"{type(exc).__name__}: {exc}") from exc
        ekin = float(work.get_kinetic_energy())
        if not np.isfinite(epot) or not np.isfinite(ekin):
            raise _MDStop(step, "non-finite energy")
        dmin = _min_distance(work)
        if np.isfinite(dmin) and dmin < settings.min_distance_floor:
            raise _MDStop(step, f"unphysical minimum distance {dmin:.3f} A")
        k = state["sample"]
        positions[k] = work.get_positions()
        cells[k] = work.cell.array
        step_l.append(step)
        time_l.append(step * settings.timestep_fs)
        epot_l.append(epot)
        ekin_l.append(ekin)
        econs_l.append(epot + ekin + thermostat_energy())
        temp_l.append(float(work.get_temperature()))
        dmin_l.append(dmin)
        com_l.append(float(np.linalg.norm(work.get_center_of_mass() - com0)))
        state["sample"] = k + 1
        if progress and step % progress_every == 0:
            progress(step, settings.n_steps)

    if settings.temperature_end_k is not None:
        dyn.attach(ramp, interval=1)
    dyn.attach(sample, interval=settings.sample_every)

    crashed_at: int | None = None
    crash_reason: str | None = None
    t0 = time.time()
    try:
        dyn.run(settings.n_steps)
    except _MDStop as stop:
        crashed_at, crash_reason = stop.step, stop.reason
    except Exception as exc:  # engine-level failure is also a finding
        crashed_at, crash_reason = dyn.get_number_of_steps(), f"{type(exc).__name__}: {exc}"
    wall = time.time() - t0
    if progress:
        progress(settings.n_steps if crashed_at is None else crashed_at, settings.n_steps)

    n = state["sample"]
    traj = Trajectory(
        symbols=list(work.get_chemical_symbols()),
        positions=positions[:n],
        cells=cells[:n],
        name=settings.protocol_id,
        protocol=settings.to_protocol(model),
        timestep_fs=settings.timestep_fs,
        sample_stride=settings.sample_every,
        energies=np.array(epot_l),
        temperatures=np.array(temp_l),
        source=f"{model.spec.name} MD",
    )
    return MDResult(
        trajectory=traj,
        settings=settings,
        step=np.array(step_l),
        time_fs=np.array(time_l),
        potential_energy=np.array(epot_l),
        kinetic_energy=np.array(ekin_l),
        total_energy=np.array(econs_l),
        temperature=np.array(temp_l),
        min_distance=np.array(dmin_l),
        com_drift=np.array(com_l),
        wall_time_s=wall,
        crashed_at=crashed_at,
        crash_reason=crash_reason,
    )


def save_md(result: MDResult, path) -> "Path":  # type: ignore[name-defined]
    """Persist a trajectory + diagnostics so notebooks need not re-run MD."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        symbols=np.array(result.trajectory.symbols),
        positions=result.trajectory.positions,
        cells=result.trajectory.cells,
        step=result.step,
        time_fs=result.time_fs,
        potential_energy=result.potential_energy,
        kinetic_energy=result.kinetic_energy,
        # 守恒量必须单独存。NVT 下它是 pe + ke + **热浴能**，而热浴能既不在
        # potential_energy 也不在 kinetic_energy 里——早先 load_md 用 pe+ke 重建，
        # 于是任何从盘上读回的 NVT run 的 drift 量的都不是守恒量，而是系统与热浴
        # 交换能量这件事本身（它本来就该起伏）。NVE 下两者恰好相等，所以 L1 的
        # NVE drift 一直是对的，只有 NVT 的诊断受影响。
        total_energy=result.total_energy,
        temperature=result.temperature,
        min_distance=result.min_distance,
        com_drift=result.com_drift,
        wall_time_s=result.wall_time_s,
        crashed_at=-1 if result.crashed_at is None else result.crashed_at,
        crash_reason=result.crash_reason or "",
        settings=repr(result.settings),
        engine=result.settings.engine,
        timestep_fs=result.settings.timestep_fs,
        sample_every=result.settings.sample_every,
        protocol_id=result.settings.protocol_id,
        equilibration_fraction=result.settings.equilibration_fraction,
        temperature_k=result.settings.temperature_k,
        temperature_end_k=(
            np.nan if result.settings.temperature_end_k is None else result.settings.temperature_end_k
        ),
        protocol_json=json.dumps(
            result.trajectory.protocol.to_dict() if result.trajectory.protocol else {}, ensure_ascii=False
        ),
    )
    return path


def load_md(path, settings: MDSettings | None = None) -> MDResult:
    """Reload a saved run, including the protocol its trajectory was produced under."""
    from pathlib import Path

    z = np.load(Path(path), allow_pickle=False)
    t_end = float(z["temperature_end_k"]) if "temperature_end_k" in z else float("nan")
    settings = settings or MDSettings(
        timestep_fs=float(z["timestep_fs"]),
        sample_every=int(z["sample_every"]),
        # 缺这一项的是字段引入之前存的盘，来源**未知**——不要猜。
        # （曾经写成"缺省即 ase"，但当天早些时候 LAMMPS 产出的文件同样缺字段，
        #  于是文件名对、内容标错，把引擎一致性检查变成了误报。）
        # 此时以文件名里的引擎为准，见 StatePoint.md_path_for。
        engine=str(z["engine"]) if "engine" in z else "",
        protocol_id=str(z["protocol_id"]),
        temperature_k=float(z["temperature_k"]) if "temperature_k" in z else 0.0,
        temperature_end_k=None if not np.isfinite(t_end) else t_end,
        equilibration_fraction=(
            float(z["equilibration_fraction"]) if "equilibration_fraction" in z else 0.5
        ),
    )
    protocol = None
    if "protocol_json" in z:
        data = json.loads(str(z["protocol_json"]))
        if data:
            allowed = {f.name for f in fields(Protocol)}
            temperature = data.get("temperature_k")
            if isinstance(temperature, list):
                data["temperature_k"] = tuple(temperature)
            protocol = Protocol(**{k: v for k, v in data.items() if k in allowed})
    traj = Trajectory(
        symbols=[str(s) for s in z["symbols"]],
        positions=z["positions"],
        cells=z["cells"],
        name=str(z["protocol_id"]),
        protocol=protocol,
        timestep_fs=float(z["timestep_fs"]),
        sample_stride=int(z["sample_every"]),
        energies=z["potential_energy"],
        temperatures=z["temperature"],
        source=str(path),
    )
    crashed = int(z["crashed_at"])
    return MDResult(
        trajectory=traj,
        settings=settings,
        step=z["step"],
        time_fs=z["time_fs"],
        potential_energy=z["potential_energy"],
        kinetic_energy=z["kinetic_energy"],
        # 2026-09-09 之前存盘的文件没有这一项，退回旧行为（pe+ke）以保持可读；
        # 那些文件的 NVT drift 读数**不是守恒量的漂移**，不要用来比较热浴质量。
        total_energy=(
            z["total_energy"] if "total_energy" in z
            else z["potential_energy"] + z["kinetic_energy"]
        ),
        temperature=z["temperature"],
        min_distance=z["min_distance"],
        com_drift=z["com_drift"],
        wall_time_s=float(z["wall_time_s"]),
        crashed_at=None if crashed < 0 else crashed,
        crash_reason=str(z["crash_reason"]) or None,
    )
