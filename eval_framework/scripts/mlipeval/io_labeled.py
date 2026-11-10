"""Reader for labelled extxyz reference sets (``REF_energy``/``REF_forces``/``Virial``).

The 615-frame GST225 set carries a ``Virial`` array whose sign, unit and
component order are *not* traceable to a recorded protocol.  This module keeps
stress optional and provides :func:`audit_virial_convention`, which reports what
each candidate convention would imply instead of silently picking one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from ase import Atoms
from ase.io import read as ase_read

from .schema import DataRole, StructureRef


@dataclass
class LabelledSet:
    """Frames with reference labels, all belonging to one ``protocol_id``."""

    frames: list[Atoms]
    protocol_id: str
    energy_key: str = "REF_energy"
    force_key: str = "REF_forces"
    virial_key: str | None = "Virial"
    role: DataRole = DataRole.DIAGNOSTIC
    source: str = ""
    notes: str = ""
    tags: dict[str, np.ndarray] = field(default_factory=dict)
    #: Per-frame stress in eV/A^3 with the ASE sign, for sets harvested from
    #: OUTCAR under a known convention.  Frames without it are left out of the
    #: stress statistics rather than voiding them for the whole set.
    stress_key: str | None = None

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def n_atoms(self) -> np.ndarray:
        return np.array([len(a) for a in self.frames])

    @property
    def energies(self) -> np.ndarray:
        return np.array([float(a.info[self.energy_key]) for a in self.frames])

    @property
    def volumes(self) -> np.ndarray:
        return np.array([a.get_volume() for a in self.frames])

    def forces(self) -> list[np.ndarray]:
        return [np.asarray(a.arrays[self.force_key], dtype=float) for a in self.frames]

    def virials(self) -> np.ndarray | None:
        if not self.virial_key:
            return None
        out = []
        for a in self.frames:
            if self.virial_key not in a.info:
                return None
            out.append(np.asarray(a.info[self.virial_key], dtype=float).reshape(3, 3))
        return np.array(out)

    def stresses(self) -> tuple[np.ndarray, np.ndarray] | None:
        """``(mask, stress)``: which frames carry ``stress_key``, and their ``(n, 3, 3)`` stress."""
        if not self.stress_key:
            return None
        mask = np.array([self.stress_key in a.info for a in self.frames])
        if not mask.any():
            return None
        stress = np.array([np.asarray(a.info[self.stress_key], dtype=float).reshape(3, 3)
                           for a, m in zip(self.frames, mask) if m])
        return mask, stress

    def config_types(self) -> np.ndarray:
        return np.array([str(a.info.get("config_type", "unknown")) for a in self.frames])

    def symbols(self) -> list[np.ndarray]:
        return [np.asarray(a.get_chemical_symbols()) for a in self.frames]

    def compositions(self) -> np.ndarray:
        return np.array([a.get_chemical_formula() for a in self.frames])

    def structure_refs(self) -> list[StructureRef]:
        refs = []
        for i, a in enumerate(self.frames):
            refs.append(
                StructureRef(
                    structure_id=f"{self.protocol_id}:{a.info.get('config_type', i)}",
                    source=self.source,
                    n_atoms=len(a),
                    composition=a.get_chemical_formula(),
                    role=self.role,
                    protocol_id=self.protocol_id,
                    frame_index=i,
                )
            )
        return refs

    def subset(self, mask: Sequence[bool] | np.ndarray) -> "LabelledSet":
        mask = np.asarray(mask, dtype=bool)
        return LabelledSet(
            frames=[f for f, m in zip(self.frames, mask) if m],
            protocol_id=self.protocol_id,
            energy_key=self.energy_key,
            force_key=self.force_key,
            virial_key=self.virial_key,
            role=self.role,
            source=self.source,
            notes=self.notes,
            tags={k: np.asarray(v)[mask] for k, v in self.tags.items()},
            stress_key=self.stress_key,
        )


def read_labelled_extxyz(
    path: str | Path,
    *,
    protocol_id: str,
    role: DataRole = DataRole.DIAGNOSTIC,
    energy_key: str = "REF_energy",
    force_key: str = "REF_forces",
    virial_key: str | None = "Virial",
    index: str = ":",
    notes: str = "",
) -> LabelledSet:
    path = Path(path)
    frames = ase_read(str(path), index=index, format="extxyz")
    if isinstance(frames, Atoms):
        frames = [frames]
    missing = [i for i, a in enumerate(frames) if energy_key not in a.info or force_key not in a.arrays]
    if missing:
        raise ValueError(f"{len(missing)} frames in {path} lack {energy_key}/{force_key}")
    have_virial = virial_key is not None and all(virial_key in a.info for a in frames)
    return LabelledSet(
        frames=frames,
        protocol_id=protocol_id,
        energy_key=energy_key,
        force_key=force_key,
        virial_key=virial_key if have_virial else None,
        role=role,
        source=str(path),
        notes=notes,
    )


# --------------------------------------------------------------------------
# config_type decoding for the GST225 615-frame set
# --------------------------------------------------------------------------
_GST_CONFIG = re.compile(r"^d(?P<density>\d+)_(?P<stage>[^_]+(?:-[^_]+)*)_(?P<step>\d+)$")

#: AIMD stage -> thermodynamic state of the 216-atom source trajectories, read
#: from each stage INCAR (TEBEG/TEEND).  Note that neither "300 K" stage is
#: isothermal: 4-MD ramps 300->200 K and 5-MD ramps 300->400 K, so neither may
#: be used as an equilibrium 300 K state point.
GST_STAGE_STATE = {
    "1-2000K": {"label": "liquid, isothermal 2000 K", "temperature_k": 2000.0, "kind": "liquid"},
    "2-300K": {"label": "quench ramp 2000 -> 300 K", "temperature_k": None, "kind": "quench"},
    "3-opt": {"label": "ionic relaxation (0 K)", "temperature_k": 0.0, "kind": "relaxed"},
    "4-MD": {"label": "amorphous, ramp 300 -> 200 K", "temperature_k": None, "kind": "amorphous_ramp"},
    "5-MD": {"label": "amorphous, ramp 300 -> 400 K", "temperature_k": None, "kind": "amorphous_ramp"},
}


def decode_gst_config_types(labelled: LabelledSet) -> dict[str, np.ndarray]:
    """Split ``d27_1-2000K_1997`` into density tag, AIMD stage and source step.

    ``d27`` etc. is the nominal number density in units of 1e-3 atom/A^3; it is
    cross-checked against the actual cell volume and reported, never assumed.
    """
    densities, stages, steps, states, temps = [], [], [], [], []
    for atoms in labelled.frames:
        raw = str(atoms.info.get("config_type", ""))
        m = _GST_CONFIG.match(raw)
        if m:
            densities.append(f"d{m.group('density')}")
            stage = m.group("stage")
            stages.append(stage)
            steps.append(int(m.group("step")))
        else:
            densities.append("unknown")
            stages.append("unknown")
            steps.append(-1)
        info = GST_STAGE_STATE.get(stages[-1], {})
        states.append(info.get("kind", "unknown"))
        temps.append(info.get("temperature_k", np.nan))

    actual = labelled.n_atoms / labelled.volumes
    tags = {
        "density_tag": np.array(densities),
        "stage": np.array(stages),
        "source_step": np.array(steps),
        "state_kind": np.array(states),
        "nominal_temperature_k": np.array(temps, dtype=float),
        "number_density": actual,
    }
    labelled.tags.update(tags)
    return tags


# --------------------------------------------------------------------------
# virial / stress convention audit
# --------------------------------------------------------------------------
def virial_to_stress(virial: np.ndarray, volume: float | np.ndarray, sign: int = -1) -> np.ndarray:
    """``stress = sign * virial / V``.

    ``sign=-1`` reproduces the ASE/VASP convention used by
    ``GST225_ref/scripts/outcars2extxyz_stress.py`` (``REF_stress = -Virial/V``).
    """
    virial = np.asarray(virial, dtype=float)
    volume = np.asarray(volume, dtype=float)
    if virial.ndim == 3:
        return sign * virial / volume[:, None, None]
    return sign * virial / float(volume)


@dataclass
class VirialAudit:
    """Result of comparing both sign conventions against a model's stress."""

    n_frames: int
    symmetric: bool
    max_asymmetry: float
    trace_mean_ev: float
    candidates: dict[str, dict[str, float]]
    recommendation: str
    verdict: str

    def to_dict(self) -> dict[str, object]:
        return {
            "n_frames": self.n_frames,
            "symmetric": self.symmetric,
            "max_asymmetry": self.max_asymmetry,
            "trace_mean_ev": self.trace_mean_ev,
            "candidates": self.candidates,
            "recommendation": self.recommendation,
            "verdict": self.verdict,
        }


def audit_virial_convention(
    labelled: LabelledSet,
    model_stress: np.ndarray | None = None,
) -> VirialAudit:
    """Report what each sign convention implies; never silently pick one.

    ``model_stress`` is ``(n_frames, 3, 3)`` in eV/A^3 (ASE sign convention).
    Without it the audit only checks internal consistency (symmetry, magnitude).
    The returned ``verdict`` is the string that must be quoted in any L0 stress
    metric that uses this label set.
    """
    virials = labelled.virials()
    if virials is None:
        return VirialAudit(0, False, np.nan, np.nan, {}, "unavailable", "no Virial field present")

    asym = np.abs(virials - np.transpose(virials, (0, 2, 1))).max()
    volumes = labelled.volumes
    candidates: dict[str, dict[str, float]] = {}
    recommendation = "undetermined"

    if model_stress is not None:
        model_stress = np.asarray(model_stress, dtype=float).reshape(len(labelled), 3, 3)
        for name, sign in (("stress = -Virial/V", -1), ("stress = +Virial/V", +1)):
            ref = virial_to_stress(virials, volumes, sign=sign)
            x, y = ref.reshape(-1), model_stress.reshape(-1)
            r = float(np.corrcoef(x, y)[0, 1])
            slope = float(np.polyfit(x, y, 1)[0])
            candidates[name] = {
                "pearson_r": r,
                "slope_model_vs_ref": slope,
                "rmse_ev_per_A3": float(np.sqrt(np.mean((x - y) ** 2))),
            }
        best = max(candidates, key=lambda k: candidates[k]["pearson_r"])
        if candidates[best]["pearson_r"] > 0.8:
            recommendation = best
        verdict = (
            f"model-vs-reference correlation favours '{recommendation}'"
            if recommendation != "undetermined"
            else "neither convention correlates with the model; do NOT use Virial for scoring"
        )
    else:
        verdict = (
            "no model stress supplied: convention unverified, Virial may only be "
            "used for shape/trend diagnostics, never for a scored stress metric"
        )

    return VirialAudit(
        n_frames=len(labelled),
        symmetric=bool(asym < 1e-6 * max(1.0, np.abs(virials).max())),
        max_asymmetry=float(asym),
        trace_mean_ev=float(np.mean(np.trace(virials, axis1=1, axis2=2)) / 3.0),
        candidates=candidates,
        recommendation=recommendation,
        verdict=verdict,
    )


# --------------------------------------------------------------------------
# leakage screening against external training sets
# --------------------------------------------------------------------------
def composition_signature(atoms: Atoms) -> str:
    return atoms.get_chemical_formula(mode="hill")


def leakage_screen(
    labelled: LabelledSet,
    external_path: str | Path,
    *,
    max_external_frames: int | None = 20000,
    rdf_bins: int = 60,
    r_max: float = 6.0,
    threshold: float = 0.02,
) -> dict[str, object]:
    """Cheap near-duplicate screen against an external training set.

    Compares a coarse total-RDF fingerprint per frame; frames whose nearest
    external neighbour is closer than ``threshold`` are flagged as *possible*
    duplicates.  This is a screen, not a proof: it exists so that a frame is
    never promoted to ``protected_audit`` without a check.
    """
    from .neighbors import total_rdf_fingerprint

    external_path = Path(external_path)
    if not external_path.exists():
        return {"status": "skipped", "reason": f"{external_path} not found"}

    ext = ase_read(str(external_path), index=f":{max_external_frames}" if max_external_frames else ":")
    if isinstance(ext, Atoms):
        ext = [ext]
    target_comp = {composition_signature(a) for a in labelled.frames}
    ext = [a for a in ext if composition_signature(a) in target_comp or len(a) == len(labelled.frames[0])]
    if not ext:
        return {
            "status": "clear",
            "reason": "no external frame shares the composition of the audit set",
            "n_external_considered": 0,
        }

    ref_fp = np.array([total_rdf_fingerprint(a, r_max=r_max, bins=rdf_bins) for a in labelled.frames])
    ext_fp = np.array([total_rdf_fingerprint(a, r_max=r_max, bins=rdf_bins) for a in ext])
    d = np.linalg.norm(ref_fp[:, None, :] - ext_fp[None, :, :], axis=-1)
    nearest = d.min(axis=1)
    flagged = np.flatnonzero(nearest < threshold)
    return {
        "status": "flagged" if flagged.size else "clear",
        "n_external_considered": len(ext),
        "nearest_distance": nearest,
        "flagged_indices": flagged.tolist(),
        "threshold": threshold,
        "note": "RDF-fingerprint screen only; a flag means 'inspect', not 'identical'",
    }
