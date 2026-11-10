"""Pinned MLIP calculator adapter.

One entry point (:func:`build_model`) returns both the ASE calculator and a
fully pinned :class:`~mlipeval.schema.ModelSpec` (checkpoint sha256, dtype,
device, seed, software versions), so every metric produced downstream can name
exactly which model produced it.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from ase import Atoms

from . import models as model_registry
from .paths import MACE_MPA0, n_threads
from .schema import ModelSpec, Protocol

# MACE-MPA-0 is trained on MPtrj + sAlex, whose native theory level is PBE+U.
# It is NOT the same protocol as this project's PBE reference; that mismatch is
# recorded here so it can be quoted in every L0 result.
MACE_MPA0_NATIVE_PROTOCOL = "PBE+U (MPtrj + sAlex), no dispersion"


def _pin_threads(threads: int | None = None) -> int:
    import torch

    t = threads or n_threads()
    torch.set_num_threads(t)
    os.environ.setdefault("OMP_NUM_THREADS", str(t))
    return t


@dataclass
class Model:
    """An ASE calculator plus its pinned identity."""

    calculator: Any
    spec: ModelSpec

    def protocol(self, protocol_id: str, **kwargs) -> Protocol:
        """A ``Protocol`` describing *the model* as a label source."""
        base = dict(
            protocol_id=protocol_id,
            kind=kwargs.pop("kind", "mlip_static"),
            description=kwargs.pop("description", f"{self.spec.name} single points"),
            code=self.spec.name,
            code_version=self.spec.software.get("mace", "unknown"),
            functional=self.spec.native_protocol,
            dispersion="none" if not self.spec.dispersion else "on",
            provenance={"checkpoint_sha256": self.spec.checkpoint_sha256, "dtype": self.spec.dtype},
        )
        base.update(kwargs)
        return Protocol(**base)

    def fresh_atoms(self, atoms: Atoms) -> Atoms:
        clone = atoms.copy()
        clone.calc = self.calculator
        return clone

    # -- convenience single-point ------------------------------------------
    def energy(self, atoms: Atoms) -> float:
        return float(self.fresh_atoms(atoms).get_potential_energy())

    def energy_forces(self, atoms: Atoms) -> tuple[float, np.ndarray]:
        clone = self.fresh_atoms(atoms)
        return float(clone.get_potential_energy()), np.asarray(clone.get_forces())

    def energy_forces_stress(self, atoms: Atoms) -> tuple[float, np.ndarray, np.ndarray]:
        clone = self.fresh_atoms(atoms)
        return (
            float(clone.get_potential_energy()),
            np.asarray(clone.get_forces()),
            np.asarray(clone.get_stress(voigt=False)),
        )


def model_spec_only(
    checkpoint: str | Path | None = None,
    *,
    name: str | None = None,
    dtype: str | None = None,
    device: str = "auto",
    seed: int | None = 0,
    native_protocol: str | None = None,
    model_key: str | None = None,
) -> ModelSpec:
    """The pinned identity of the model, without building the network.

    Cache lookups only need the checkpoint hash, so a notebook that just reads
    precomputed results should not pay for loading the network (~11 s on CPU for
    MACE) or, worse, end up evaluating anything on the login node.

    With no arguments this describes whichever model ``MLIPEVAL_MODEL`` selects.
    """
    from .pbs import select_device

    entry = model_registry.get(model_key or model_registry.current_key())
    spec = ModelSpec.from_checkpoint(
        Path(checkpoint) if checkpoint is not None else entry.checkpoint,
        name=name or entry.label,
        dtype=dtype or entry.dtype,
        device=select_device(device),
        seed=seed,
        native_protocol=native_protocol or entry.native_protocol,
    )
    spec.software["artifact"] = entry.loadable.name
    return spec


# --------------------------------------------------------------------------
# per-backend calculator construction
# --------------------------------------------------------------------------
def _build_mace(entry, device: str, dtype: str):
    from mace.calculators import MACECalculator

    return MACECalculator(model_paths=str(entry.loadable), device=device, default_dtype=dtype)


def _build_nequip(entry, device: str, dtype: str):
    """Eager load straight from the ``.nequip.zip`` package archive.

    ``nequip-compile``/``from_compiled_model`` is the faster route, but an
    AOTInductor artefact is pinned to one device *and* to the shapes it was
    traced with, while L0 alone spans three cell sizes.  Correctness of the
    numbers matters more here than inference speed, so the packaged model is
    loaded as-is.
    """
    from nequip.ase import NequIPCalculator

    return NequIPCalculator._from_packaged_model(
        package_path=str(entry.loadable), device=device
    )


def _build_deepmd(entry, device: str, dtype: str):
    from deepmd.calculator import DP

    return DP(model=str(entry.loadable))


def _build_mattersim(entry, device: str, dtype: str):
    from mattersim.forcefield import MatterSimCalculator

    return MatterSimCalculator(
        load_path=str(entry.loadable), compute_stress=True, device=device
    )


def _build_chgnet(entry, device: str, dtype: str):
    from chgnet.model import CHGNet, CHGNetCalculator

    return CHGNetCalculator(model=CHGNet.from_file(str(entry.loadable)), use_device=device)


_BUILDERS = {
    "mace": _build_mace,
    "nequip": _build_nequip,
    "deepmd": _build_deepmd,
    "mattersim": _build_mattersim,
    "chgnet": _build_chgnet,
}


def build_model(
    checkpoint: str | Path | None = None,
    *,
    name: str | None = None,
    device: str = "auto",
    dtype: str | None = None,
    seed: int | None = 0,
    threads: int | None = None,
    dispersion: bool = False,
    native_protocol: str | None = None,
    model_key: str | None = None,
) -> Model:
    """Load the selected model with every knob that changes a number pinned.

    Which model is loaded comes from ``MLIPEVAL_MODEL`` (see
    :mod:`mlipeval.models`); each backend needs its own conda environment, so a
    driver is expected to be running inside that environment already.

    ``device="auto"`` resolves to CUDA whenever a GPU is visible.  That is the
    normal case inside a PBS job on a GPU node; on the login node there is no
    CUDA at all, and heavy drivers refuse to run there anyway
    (:func:`mlipeval.pbs.require_compute_node`).
    """
    import torch

    from .pbs import select_device

    entry = model_registry.get(model_key or model_registry.current_key())
    device = select_device(device)
    dtype = dtype or entry.dtype
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    _pin_threads(threads)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    torch.use_deterministic_algorithms(False)  # no deterministic scatter on CPU

    calculator = _BUILDERS[entry.backend](entry, device, dtype)

    spec = ModelSpec.from_checkpoint(
        Path(checkpoint) if checkpoint is not None else entry.checkpoint,
        name=name or entry.label,
        dtype=dtype,
        device=device,
        seed=seed,
        native_protocol=native_protocol or entry.native_protocol,
        dispersion=dispersion,
    )
    # The artefact actually opened is part of the model's identity: a frozen or
    # compiled export of the same checkpoint does not have to give the same
    # numbers, and the prediction cache is keyed on this.
    spec.software["artifact"] = entry.loadable.name
    return Model(calculator=calculator, spec=spec)


# --------------------------------------------------------------------------
# batch evaluation
# --------------------------------------------------------------------------
@dataclass
class Predictions:
    """Model predictions over a frame set, aligned with the input order."""

    energies: np.ndarray
    forces: list[np.ndarray]
    stresses: np.ndarray | None
    n_atoms: np.ndarray
    volumes: np.ndarray
    wall_time_s: float
    failures: list[tuple[int, str]]

    def __len__(self) -> int:
        return len(self.energies)

    @property
    def energy_per_atom(self) -> np.ndarray:
        return self.energies / self.n_atoms

    def flat_forces(self) -> np.ndarray:
        return np.concatenate([f.reshape(-1) for f in self.forces]) if self.forces else np.zeros(0)


def predict(
    frames: Sequence[Atoms],
    model: Model,
    *,
    with_stress: bool = True,
    progress: Callable[[int, int], None] | None = None,
    progress_every: int = 25,
) -> Predictions:
    """Evaluate a frame list, recording failures instead of aborting."""
    energies = np.full(len(frames), np.nan)
    volumes = np.full(len(frames), np.nan)
    n_atoms = np.zeros(len(frames), dtype=int)
    forces: list[np.ndarray] = []
    stresses = np.full((len(frames), 3, 3), np.nan) if with_stress else None
    failures: list[tuple[int, str]] = []

    t0 = time.time()
    for k, frame in enumerate(frames):
        n_atoms[k] = len(frame)
        volumes[k] = frame.get_volume()
        try:
            clone = model.fresh_atoms(frame)
            energies[k] = float(clone.get_potential_energy())
            forces.append(np.asarray(clone.get_forces(), dtype=float))
            if with_stress:
                stresses[k] = np.asarray(clone.get_stress(voigt=False), dtype=float)
        except Exception as exc:  # a model failure is data, not a crash
            forces.append(np.full((len(frame), 3), np.nan))
            failures.append((k, f"{type(exc).__name__}: {exc}"))
        if progress and (k % progress_every == 0 or k == len(frames) - 1):
            progress(k + 1, len(frames))

    return Predictions(
        energies=energies,
        forces=forces,
        stresses=stresses,
        n_atoms=n_atoms,
        volumes=volumes,
        wall_time_s=time.time() - t0,
        failures=failures,
    )


def descriptors(frames: Sequence[Atoms], model: Model, *, invariants_only: bool = True) -> np.ndarray:
    """Per-structure MACE descriptors (mean over atoms).

    Used by the failure-export stage to deduplicate candidate structures without
    inventing a separate fingerprint.
    """
    if not hasattr(model.calculator, "get_descriptors"):
        raise NotImplementedError(
            f"{model.spec.name} exposes no descriptor API; the candidate-export and "
            "bridge-set stages are MACE-only and are skipped for this model."
        )
    out = []
    for frame in frames:
        d = model.calculator.get_descriptors(frame.copy(), invariants_only=invariants_only)
        out.append(np.asarray(d, dtype=float).mean(axis=0))
    return np.array(out)


def atom_descriptors(frame: Atoms, model: Model, *, invariants_only: bool = True) -> np.ndarray:
    if not hasattr(model.calculator, "get_descriptors"):
        raise NotImplementedError(f"{model.spec.name} exposes no descriptor API")
    return np.asarray(model.calculator.get_descriptors(frame.copy(), invariants_only=invariants_only), dtype=float)


def stdout_progress(prefix: str = "") -> Callable[[int, int], None]:
    start = time.time()

    def _report(done: int, total: int) -> None:
        elapsed = time.time() - start
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else float("nan")
        print(
            f"\r{prefix}{done}/{total}  {rate:.2f} frame/s  ETA {eta/60:.1f} min",
            end="" if done < total else "\n",
            flush=True,
        )

    return _report
