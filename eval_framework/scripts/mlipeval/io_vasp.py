"""Readers for the read-only VASP reference data.

``ase.io.read`` is correct but far too slow for 3000-frame XDATCARs, so the
trajectory reader here is a purpose-built streaming parser.  Everything returns
plain numpy plus a :class:`~mlipeval.schema.Protocol` recovered from the INCAR,
so that no analysis silently loses the protocol it belongs to.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from ase import Atoms

from .schema import Protocol

# --------------------------------------------------------------------------
# INCAR
# --------------------------------------------------------------------------
def read_incar(path: str | Path) -> dict[str, str]:
    """Parse an INCAR into an upper-cased tag dictionary."""
    tags: dict[str, str] = {}
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split("!", 1)[0].split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().upper()
        if key:
            tags[key] = value.strip()
    return tags


def _f(tags: dict[str, str], key: str) -> float | None:
    if key not in tags:
        return None
    token = tags[key].split()[0].replace("D", "E").rstrip(",;")
    try:
        return float(token)
    except ValueError:
        return None


def _i(tags: dict[str, str], key: str) -> int | None:
    value = _f(tags, key)
    return None if value is None else int(value)


def protocol_from_incar(
    path: str | Path,
    protocol_id: str,
    *,
    kind: str | None = None,
    description: str = "",
    **overrides,
) -> Protocol:
    """Recover a :class:`Protocol` from a VASP INCAR.

    Fields VASP does not record (functional family, PAW set, code version) stay
    ``None`` unless passed via ``overrides`` -- they must never be guessed.
    """
    path = Path(path)
    tags = read_incar(path)
    ibrion = _i(tags, "IBRION")
    nsw = _i(tags, "NSW")
    mdalgo = _i(tags, "MDALGO")
    if kind is None:
        if ibrion == 0:
            kind = "aimd"
        elif nsw and nsw > 0:
            kind = "relaxation"
        else:
            kind = "dft_static"

    thermostat = None
    ensemble = None
    if ibrion == 0:
        if mdalgo == 2:
            thermostat, ensemble = "Nose-Hoover (SMASS=%s)" % tags.get("SMASS", "?"), "NVT"
        elif mdalgo == 1:
            prob = _f(tags, "ANDERSEN_PROB")
            if prob is not None and prob == 0.0:
                thermostat, ensemble = "none (ANDERSEN_PROB=0)", "NVE"
            else:
                thermostat, ensemble = "Andersen", "NVT"
        elif mdalgo is not None:
            thermostat, ensemble = f"MDALGO={mdalgo}", "MD"
        else:
            # Legacy VASP MD without MDALGO: the ensemble is selected by SMASS.
            smass = _f(tags, "SMASS")
            if smass is None:
                thermostat, ensemble = "unspecified", "MD"
            elif smass == -3:
                thermostat, ensemble = "none (SMASS=-3)", "NVE"
            elif smass == -2:
                thermostat, ensemble = "none (SMASS=-2, velocities kept)", "NVE"
            elif smass == -1:
                thermostat, ensemble = "velocity scaling (SMASS=-1)", "scaled-T MD (not a proper ensemble)"
            else:
                thermostat, ensemble = f"Nose-Hoover (SMASS={smass:g})", "NVT"

    tebeg, teend = _f(tags, "TEBEG"), _f(tags, "TEEND")
    if tebeg is not None and teend is not None and teend != tebeg:
        temperature: float | tuple[float, float] | None = (tebeg, teend)
    else:
        temperature = tebeg

    kpoints = None
    if "KSPACING" in tags:
        kpoints = f"KSPACING={tags['KSPACING']}"
    elif (path.parent / "KPOINTS").exists():
        kpoints = " ".join((path.parent / "KPOINTS").read_text().split("\n")[3].split())

    fields = dict(
        protocol_id=protocol_id,
        kind=kind,
        description=description,
        code="VASP",
        encut_ev=_f(tags, "ENCUT"),
        kpoints=kpoints,
        smearing=f"ISMEAR={tags.get('ISMEAR', '?')}, SIGMA={tags.get('SIGMA', '?')}",
        ediff_ev=_f(tags, "EDIFF"),
        ensemble=ensemble,
        thermostat=thermostat,
        timestep_fs=_f(tags, "POTIM") if ibrion == 0 else None,
        n_steps=nsw,
        temperature_k=temperature,
        provenance={"incar": str(path), "tags": tags},
    )
    fields.update(overrides)
    return Protocol(**fields)


# --------------------------------------------------------------------------
# trajectories
# --------------------------------------------------------------------------
@dataclass
class Trajectory:
    """Fixed-composition trajectory held as plain arrays.

    ``positions`` are Cartesian, shape ``(n_frames, n_atoms, 3)``.  ``cells`` is
    ``(n_frames, 3, 3)`` so that a variable-cell run needs no special casing.
    """

    symbols: list[str]
    positions: np.ndarray
    cells: np.ndarray
    name: str = ""
    protocol: Protocol | None = None
    timestep_fs: float | None = None
    sample_stride: int = 1
    energies: np.ndarray | None = None
    temperatures: np.ndarray | None = None
    source: str = ""

    def __post_init__(self) -> None:
        self.positions = np.ascontiguousarray(self.positions, dtype=float)
        self.cells = np.ascontiguousarray(self.cells, dtype=float)
        if self.cells.ndim == 2:
            self.cells = np.repeat(self.cells[None], len(self.positions), axis=0)
        if self.positions.shape[1] != len(self.symbols):
            raise ValueError("positions/symbols mismatch")

    # -- basic properties --------------------------------------------------
    def __len__(self) -> int:
        return int(self.positions.shape[0])

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)

    @property
    def species(self) -> list[str]:
        seen: list[str] = []
        for s in self.symbols:
            if s not in seen:
                seen.append(s)
        return seen

    @property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.cells[0])))

    @property
    def number_density(self) -> float:
        return self.n_atoms / self.volume

    @property
    def frame_time_fs(self) -> np.ndarray:
        dt = (self.timestep_fs or 1.0) * self.sample_stride
        return np.arange(len(self)) * dt

    def indices_of(self, species: str) -> np.ndarray:
        return np.flatnonzero(np.asarray(self.symbols) == species)

    # -- slicing -----------------------------------------------------------
    def slice(self, start: int = 0, stop: int | None = None, stride: int = 1) -> "Trajectory":
        sel = slice(start, stop, stride)
        return Trajectory(
            symbols=list(self.symbols),
            positions=self.positions[sel],
            cells=self.cells[sel],
            name=f"{self.name}[{start}:{stop}:{stride}]",
            protocol=self.protocol,
            timestep_fs=self.timestep_fs,
            sample_stride=self.sample_stride * stride,
            energies=None if self.energies is None else self.energies[sel],
            temperatures=None if self.temperatures is None else self.temperatures[sel],
            source=self.source,
        )

    def last_fraction(self, fraction: float = 0.5) -> "Trajectory":
        """Production segment: the trailing ``fraction`` of the trajectory."""
        start = int(len(self) * (1.0 - fraction))
        return self.slice(start=start)

    def atoms(self, index: int) -> Atoms:
        return Atoms(
            symbols=self.symbols,
            positions=self.positions[index],
            cell=self.cells[index],
            pbc=True,
        )

    def iter_atoms(self, stride: int = 1) -> Iterator[Atoms]:
        for i in range(0, len(self), stride):
            yield self.atoms(i)

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "n_frames": len(self),
            "n_atoms": self.n_atoms,
            "composition": "".join(f"{s}{self.symbols.count(s)}" for s in self.species),
            "volume_A3": round(self.volume, 4),
            "number_density_at_per_A3": round(self.number_density, 6),
            "timestep_fs": self.timestep_fs,
            "sample_stride": self.sample_stride,
            "protocol_id": self.protocol.protocol_id if self.protocol else None,
            "source": self.source,
        }


def concatenate(trajectories: "list[Trajectory]", *, name: str, protocol: Protocol | None = None) -> Trajectory:
    """Join segments of one composition and sampling into a single trajectory.

    Used for continuation segments (950 K equilibration + production) and for
    pooling independent replicas.  The time axis is only meaningful for the
    former; pooled replicas are for frame statistics such as RDF minima, never
    for anything that differentiates in time.
    """
    first = trajectories[0]
    for t in trajectories[1:]:
        if t.symbols != first.symbols or t.timestep_fs != first.timestep_fs or t.sample_stride != first.sample_stride:
            raise ValueError(f"cannot concatenate {t.name} onto {first.name}: symbols or sampling differ")

    def joined(attr: str) -> np.ndarray | None:
        arrays = [getattr(t, attr) for t in trajectories]
        return None if any(a is None for a in arrays) else np.concatenate(arrays)

    return Trajectory(
        symbols=list(first.symbols),
        positions=np.concatenate([t.positions for t in trajectories]),
        cells=np.concatenate([t.cells for t in trajectories]),
        name=name,
        protocol=protocol or first.protocol,
        timestep_fs=first.timestep_fs,
        sample_stride=first.sample_stride,
        energies=joined("energies"),
        temperatures=joined("temperatures"),
        source=" + ".join(t.source for t in trajectories),
    )


def read_xdatcar(
    path: str | Path,
    *,
    name: str = "",
    protocol: Protocol | None = None,
    timestep_fs: float | None = None,
    stride: int = 1,
    max_frames: int | None = None,
) -> Trajectory:
    """Streaming XDATCAR reader (handles fixed and variable cell)."""
    path = Path(path)
    with open(path, "r", errors="replace") as handle:
        lines = handle.read().splitlines()

    def parse_header(i: int) -> tuple[np.ndarray, list[str], int]:
        scale = float(lines[i + 1].split()[0])
        cell = np.array([[float(x) for x in lines[i + 2 + k].split()[:3]] for k in range(3)])
        cell *= scale
        names = lines[i + 5].split()
        counts = [int(x) for x in lines[i + 6].split()]
        symbols = [s for s, c in zip(names, counts) for _ in range(c)]
        return cell, symbols, i + 7

    cell0, symbols, cursor = parse_header(0)
    n_atoms = len(symbols)

    positions: list[np.ndarray] = []
    cells: list[np.ndarray] = []
    current_cell = cell0
    frame_index = -1
    i = cursor
    n_lines = len(lines)
    while i < n_lines:
        line = lines[i]
        if "configuration=" in line:
            direct = "direct" in line.lower()
            frame_index += 1
            block_start = i + 1
            if block_start + n_atoms > n_lines:
                break  # truncated final frame
            take = (frame_index % stride == 0)
            if take:
                block = lines[block_start : block_start + n_atoms]
                coords = np.fromstring(" ".join(block), sep=" ")
                if coords.size < 3 * n_atoms:
                    break
                coords = coords[: 3 * n_atoms].reshape(n_atoms, 3)
                positions.append(coords @ current_cell if direct else coords)
                cells.append(current_cell)
                if max_frames is not None and len(positions) >= max_frames:
                    break
            i = block_start + n_atoms
            continue
        # a repeated header means the cell changed (variable-cell run)
        if line.strip() and not line.strip()[0].isdigit() and i + 6 < n_lines:
            try:
                current_cell, _, i = parse_header(i)
                continue
            except (ValueError, IndexError):
                pass
        i += 1

    if not positions:
        raise ValueError(f"no frames parsed from {path}")

    return Trajectory(
        symbols=symbols,
        positions=np.array(positions),
        cells=np.array(cells),
        name=name or path.name,
        protocol=protocol,
        timestep_fs=timestep_fs if timestep_fs is not None else (protocol.timestep_fs if protocol else None),
        sample_stride=stride,
        source=str(path),
    )


def read_vasp_md_dir(
    directory: str | Path,
    *,
    protocol_id: str,
    name: str = "",
    stride: int = 1,
    max_frames: int | None = None,
    with_energies: bool = True,
) -> Trajectory:
    """Read ``INCAR`` + ``XDATCAR`` (+ ``OSZICAR``) from one AIMD directory."""
    directory = task_output_dir(directory)
    protocol = protocol_from_incar(directory / "INCAR", protocol_id, kind="aimd")
    traj = read_xdatcar(
        directory / "XDATCAR",
        name=name or directory.name,
        protocol=protocol,
        stride=stride,
        max_frames=max_frames,
    )
    oszicar = directory / "OSZICAR"
    if with_energies and oszicar.exists():
        energies, temperatures = read_oszicar_md(oszicar)
        n = len(traj)
        sel = slice(0, None, stride)
        if energies.size:
            traj.energies = energies[sel][:n]
        if temperatures.size:
            traj.temperatures = temperatures[sel][:n]
    return traj


_OSZ_MD = re.compile(r"T=\s*(\S+)\s+E=\s*(\S+)\s+F=\s*(\S+)\s+E0=\s*(\S+)")


def read_oszicar_md(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """``(E0 free energy, temperature)`` per ionic step from an MD OSZICAR."""
    energies: list[float] = []
    temperatures: list[float] = []
    for line in Path(path).read_text(errors="replace").splitlines():
        m = _OSZ_MD.search(line)
        if m:
            temperatures.append(float(m.group(1)))
            energies.append(float(m.group(4)))
    return np.array(energies), np.array(temperatures)


_ETOTAL = re.compile(r"total energy\s+ETOTAL\s*=\s*(-?\d+\.\d+)")
_EKIN = re.compile(r"kinetic energy EKIN\s*=\s*(-?\d+\.\d+)")
_TOTEN = re.compile(r"free  energy   TOTEN\s*=\s*(-?\d+\.\d+)")


def read_outcar_md_energies(path: str | Path) -> dict[str, np.ndarray]:
    """Conserved-quantity bookkeeping from an AIMD OUTCAR.

    ``ETOTAL`` is VASP's conserved quantity (potential + kinetic + thermostat
    terms); for an NVE run it is the quantity whose drift matters.
    """
    text = Path(path).read_text(errors="replace")
    return {
        "toten": np.array([float(m) for m in _TOTEN.findall(text)]),
        "ekin": np.array([float(m) for m in _EKIN.findall(text)]),
        "etotal": np.array([float(m) for m in _ETOTAL.findall(text)]),
    }


def read_outcar_final(path: str | Path) -> dict[str, object]:
    """Final energy / forces / stress / cell of a converged static or relaxation.

    ``stress`` is returned in eV/A^3 with the same sign convention as
    ``ase``: ``stress = -virial / V``.
    """
    from ase.io import read as ase_read

    atoms = ase_read(str(path), index=-1, format="vasp-out")
    out: dict[str, object] = {
        "atoms": atoms,
        "energy": float(atoms.get_potential_energy()),
        "forces": np.asarray(atoms.get_forces()),
        "volume": float(atoms.get_volume()),
        "cell": np.asarray(atoms.cell.array),
        "n_atoms": len(atoms),
    }
    try:
        out["stress"] = np.asarray(atoms.get_stress(voigt=False))
    except Exception:
        out["stress"] = None
    return out


def read_poscar(path: str | Path) -> Atoms:
    from ase.io import read as ase_read

    return ase_read(str(path), format="vasp")


# --------------------------------------------------------------------------
# where a phase-1 task actually keeps its VASP output
# --------------------------------------------------------------------------
#: Marker written by the phase-1 ``20260831_execution_v1`` release to publish the
#: one run whose step count was validated against the task's INCAR.  A task may
#: hold several attempts under ``runs/``; only this one is scientifically
#: accepted, so it is the *only* thing we follow.
_RESULT_RUN = "RESULT_RUN"


def task_output_dir(task_dir: str | Path) -> Path:
    """The directory holding a phase-1 task's accepted VASP output.

    Tasks submitted before the ``20260831_execution_v1`` execution release write
    ``OUTCAR``/``XDATCAR`` straight into the task directory.  Tasks submitted
    after it run inside a unique ``runs/run_<id>/`` directory and publish the
    accepted attempt by symlinking ``RESULT_RUN`` at it.  Both layouts are live
    in ``phase1_vasp_inputs_20260828`` simultaneously, so every reader has to go
    through here rather than assuming ``<task>/OUTCAR``.

    ``RESULT_RUN`` is followed in preference to a stale flat ``OUTCAR``: when a
    task was re-run under the new release, the published run is the accepted
    result and the flat files are the superseded first pass.  ``ACTIVE_RUN`` is
    deliberately *not* consulted -- it points at a run that may still be writing
    and has not passed step validation.
    """
    task_dir = Path(task_dir)
    published = task_dir / _RESULT_RUN
    if published.is_dir():
        return published.resolve()
    return task_dir


#: The final timing block can be followed by a long profiling dump, so the tail
#: window has to be generous -- 8 kB is not enough for a VASP 6.3 run.
_OUTCAR_TAIL_BYTES = 1 << 19


def outcar_is_complete(path: str | Path, *, tail_bytes: int = _OUTCAR_TAIL_BYTES) -> bool:
    """``True`` when VASP wrote its final timing block."""
    path = Path(path)
    if not path.exists():
        return False
    with open(path, "rb") as handle:
        handle.seek(max(0, path.stat().st_size - tail_bytes))
        tail = handle.read().decode("utf-8", errors="replace")
    return "General timing and accounting informations" in tail
