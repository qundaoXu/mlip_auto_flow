"""From diagnosis to a de-duplicated list of structures to label next.

``validation_methods_L0-L5.md`` section 6 fixes what an exported candidate must
carry: the failure it came from, why it was selected, its dedup/cluster
information, the suggested DFT protocol, and the metric IDs that must be
re-checked after the new data is added.  :class:`Candidate` is exactly that
record, and :func:`export_candidates` writes them as extxyz + a JSON manifest.

Two rules are enforced in code rather than left to discipline:

* a frame whose role is ``protected_audit`` can never become a candidate;
* de-duplication uses the model's own descriptors, so "new" means new to the
  model, not merely new to the file listing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from ase import Atoms

from .cache import cached_descriptors
from .calculators import Model
from .schema import DataRole, MetricResult, Status, _jsonable

#: Diagnosis -> what to add, transcribed from the table in section 6.
REMEDY_TABLE: dict[str, dict[str, str]] = {
    "L1.pec": {
        "structures": "short-range two-body scans for the affected pair, compressed cells, hot outlier frames",
        "protocol": "gst225_pbe_canonical_v1 static (ENCUT 500, KSPACING 0.20)",
    },
    "L1.stability": {
        "structures": "high-temperature frames just before the instability, plus compressed variants",
        "protocol": "gst225_pbe_canonical_v1 static",
    },
    "L0.force": {
        "structures": "cluster representatives at the offending element, coordination and force magnitude",
        "protocol": "gst225_pbe_canonical_v1 static",
    },
    "L0.energy": {
        "structures": "cluster representatives spanning the offending configuration type",
        "protocol": "gst225_pbe_canonical_v1 static",
    },
    "L2.eos": {
        "structures": "cubic/hexagonal strained cells and fixed-volume ionic-relaxed structures",
        "protocol": "fixed-volume ionic relaxation (H2-style) then static label",
    },
    "L2.phase_order": {
        "structures": "both phases at their DFT and MLIP relaxed geometries",
        "protocol": "gst225_pbe_canonical_v1 static",
    },
    "L3a": {
        "structures": "independent liquid/amorphous frames at the same temperature and density",
        "protocol": "gst225_pbe_canonical_v1 static on decorrelated frames",
    },
    "L3b": {
        "structures": "structures containing the key four-rings, long/short bond alternation and crystallisation precursors",
        "protocol": "gst225_pbe_canonical_v1 static",
    },
    "L4": {
        "structures": "hopping intermediates, supercooled liquid and neighbouring-temperature equilibrium frames",
        "protocol": "fixed-temperature AIMD production segment + static relabelling",
    },
    "L5": {
        "structures": "continuous frames bracketing the first divergence, not only the final structure",
        "protocol": "gst225_pbe_canonical_v1 static on a continuous window",
    },
}


def remedy_for(metric_id: str) -> dict[str, str]:
    for prefix, remedy in REMEDY_TABLE.items():
        if metric_id.startswith(prefix):
            return remedy
    return {"structures": "representative frames from the failing environment", "protocol": "gst225_pbe_canonical_v1 static"}


@dataclass
class Candidate:
    """One structure proposed for the next DFT round."""

    candidate_id: str
    source_metric: str
    level: str
    reason: str
    selection_rationale: str
    suggested_structures: str
    suggested_protocol: str
    regression_metric_ids: list[str]
    origin: dict[str, object]
    cluster_id: int = -1
    cluster_size: int = 1
    novelty: float = float("nan")
    role_of_origin: str = DataRole.DIAGNOSTIC.value
    window_group: str | None = None  # members of a group are kept or dropped together

    def to_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))


@dataclass
class CandidateSet:
    candidates: list[Candidate]
    structures: list[Atoms]
    n_before_dedup: int
    dedup_method: str
    excluded_protected: int = 0

    def __len__(self) -> int:
        return len(self.candidates)

    def manifest(self) -> dict[str, object]:
        return {
            "n_candidates": len(self.candidates),
            "n_before_dedup": self.n_before_dedup,
            "dedup_method": self.dedup_method,
            "excluded_protected_audit_frames": self.excluded_protected,
            "candidates": [c.to_dict() for c in self.candidates],
        }


# --------------------------------------------------------------------------
# collection
# --------------------------------------------------------------------------
def collect_from_l0(
    report,
    *,
    labelled,
    n_frames: int = 10,
) -> tuple[list[Candidate], list[Atoms]]:
    """Worst-error frames from an :class:`~mlipeval.l0_reference.L0Report`."""
    candidates: list[Candidate] = []
    structures: list[Atoms] = []
    worst = report.worst
    excluded = 0
    for rank, (index, score) in enumerate(zip(worst.frame_indices[:n_frames], worst.frame_scores[:n_frames])):
        if labelled.role is DataRole.PROTECTED_AUDIT:
            excluded += 1
            continue
        atoms = labelled.frames[int(index)].copy()
        remedy = remedy_for("L0.force")
        candidates.append(
            Candidate(
                candidate_id=f"L0_worst_{rank:02d}",
                source_metric="L0.force_max",
                level="L0",
                reason=f"{worst.frame_reason} = {score:.3f} eV/A",
                selection_rationale=f"rank {rank + 1} of the highest force-error frames in the labelled set",
                suggested_structures=remedy["structures"],
                suggested_protocol=remedy["protocol"],
                regression_metric_ids=["L0.force_mae", "L0.force_p99", "L0.force_max"],
                origin={
                    "frame_index": int(index),
                    "config_type": str(atoms.info.get("config_type", "")),
                    "source": labelled.source,
                    "protocol_id": labelled.protocol_id,
                },
                role_of_origin=labelled.role.value,
            )
        )
        structures.append(atoms)
    return candidates, structures


def collect_from_metrics(
    results: Iterable[MetricResult],
    structure_provider: Mapping[str, Atoms] | None = None,
) -> list[Candidate]:
    """Turn every FAILED metric into a candidate request.

    A failure without an attached structure still produces a record: the manifest
    then says which environment must be sampled, even when the frame itself has
    to be generated rather than extracted.
    """
    candidates: list[Candidate] = []
    for result in results:
        if result.status is not Status.FAILED:
            continue
        remedy = remedy_for(result.metric_id)
        candidates.append(
            Candidate(
                candidate_id=f"{result.level}_{result.metric_id.replace('.', '_')}",
                source_metric=result.metric_id,
                level=result.level,
                reason=result.failure_reason or "metric marked FAILED",
                selection_rationale="diagnostic failure requires new reference data in this environment",
                suggested_structures=remedy["structures"],
                suggested_protocol=remedy["protocol"],
                regression_metric_ids=[result.metric_id],
                origin={"structure_ids": result.structure_ids, "value": _jsonable(result.value)},
            )
        )
    return candidates


def collect_from_l5(report, trajectory, *, window: int = 3) -> tuple[list[Candidate], list[Atoms]]:
    """Continuous frames bracketing the first pathway divergence."""
    candidates: list[Candidate] = []
    structures: list[Atoms] = []
    for observable, divergence in report.divergences.items():
        if divergence.get("index") is None:
            continue
        centre_sample = int(divergence["index"])
        centre_frame = int(report.trace_ref.frame_index[centre_sample])
        remedy = remedy_for("L5")
        lo = max(0, centre_frame - window)
        hi = min(len(trajectory), centre_frame + window + 1)
        for frame_index in range(lo, hi):
            candidates.append(
                Candidate(
                    window_group=f"L5:{observable}",
                    candidate_id=f"L5_{observable}_f{frame_index:05d}",
                    source_metric=f"L5.pathway.{observable}.first_divergence_progress",
                    level="L5",
                    reason=(
                        f"{observable} leaves the reference band at progress "
                        f"{divergence['progress']:.2f}"
                    ),
                    selection_rationale=(
                        f"continuous window +-{window} frames around the first divergence, "
                        "so the departure itself is labelled rather than only its outcome"
                    ),
                    suggested_structures=remedy["structures"],
                    suggested_protocol=remedy["protocol"],
                    regression_metric_ids=[
                        f"L5.pathway.{observable}.first_divergence_progress",
                        f"L5.pathway.{observable}.final_state_error",
                    ],
                    origin={"frame_index": frame_index, "trajectory": trajectory.name},
                )
            )
            structures.append(trajectory.atoms(frame_index))
    return candidates, structures


# --------------------------------------------------------------------------
# de-duplication
# --------------------------------------------------------------------------
def deduplicate(
    candidates: Sequence[Candidate],
    structures: Sequence[Atoms],
    model: Model,
    *,
    n_clusters: int | None = None,
    min_distance: float = 0.05,
    seed: int = 0,
    preserve_windows: bool = True,
) -> CandidateSet:
    """Cluster candidates in MACE descriptor space and keep one per cluster.

    With ``n_clusters`` given, k-means picks the representatives; otherwise a
    greedy farthest-point pass keeps everything further than ``min_distance``
    from an already-kept structure.  ``novelty`` records the descriptor distance
    to the nearest kept structure, which is what makes "we already have this"
    auditable.

    ``preserve_windows`` protects continuous-frame groups.  L5 deliberately
    exports the frames *around* a divergence, and consecutive MD frames are
    nearly identical in descriptor space -- plain de-duplication would collapse
    the whole window to one frame and destroy exactly the information the
    framework asks for.  Grouped candidates are therefore de-duplicated at the
    group level: whole windows compete with each other, and a surviving window
    keeps all of its frames.
    """
    if not structures:
        return CandidateSet(list(candidates), [], len(candidates), "none (no structures attached)")

    if preserve_windows and any(c.window_group for c in candidates):
        return _deduplicate_grouped(candidates, structures, model, min_distance=min_distance)

    try:
        features = cached_descriptors(structures, model, tag="failure_export")
        features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)
        space = "MACE invariant descriptors"
    except NotImplementedError:
        # only MACE exposes descriptors; every other model is de-duplicated on a
        # model-independent fingerprint instead of not at all
        features, space = _fingerprints(structures), "total-RDF fingerprint + composition"

    if n_clusters and n_clusters < len(structures):
        from sklearn.cluster import KMeans

        labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(features)
        keep: list[int] = []
        for cluster in range(n_clusters):
            members = np.flatnonzero(labels == cluster)
            centroid = features[members].mean(axis=0)
            keep.append(int(members[np.argmin(np.linalg.norm(features[members] - centroid, axis=1))]))
        method = f"k-means(k={n_clusters}) on {space}"
    else:
        labels = np.full(len(structures), -1)
        keep = []
        for index in range(len(structures)):
            if not keep:
                keep.append(index)
                labels[index] = 0
                continue
            distance = np.linalg.norm(features[keep] - features[index], axis=1)
            nearest = int(np.argmin(distance))
            if distance[nearest] < min_distance:
                labels[index] = labels[keep[nearest]]
            else:
                labels[index] = len(keep)
                keep.append(index)
        method = f"greedy farthest-point on {space}, min distance {min_distance}"

    kept_candidates: list[Candidate] = []
    kept_structures: list[Atoms] = []
    for cluster_rank, index in enumerate(sorted(keep)):
        candidate = candidates[index]
        members = int(np.sum(labels == labels[index]))
        others = np.delete(np.arange(len(structures)), index)
        novelty = float(np.min(np.linalg.norm(features[others] - features[index], axis=1))) if others.size else np.nan
        candidate.cluster_id = int(labels[index])
        candidate.cluster_size = members
        candidate.novelty = novelty
        kept_candidates.append(candidate)
        kept_structures.append(structures[index])

    return CandidateSet(
        candidates=kept_candidates,
        structures=kept_structures,
        n_before_dedup=len(candidates),
        dedup_method=method,
    )


def _fingerprints(structures: Sequence[Atoms]) -> np.ndarray:
    """Total-RDF fingerprint with the reduced composition appended far apart.

    The same fingerprint and the same 0.02 duplicate scale as the leakage
    screens; the composition block keeps two different compositions from ever
    being merged however alike their RDFs are.
    """
    from .neighbors import total_rdf_fingerprint

    comps = sorted({a.get_chemical_formula(mode="hill", empirical=True) for a in structures})
    rows = []
    for atoms in structures:
        onehot = np.zeros(len(comps))
        onehot[comps.index(atoms.get_chemical_formula(mode="hill", empirical=True))] = 10.0
        rows.append(np.concatenate([total_rdf_fingerprint(atoms, r_max=6.0, bins=60), onehot]))
    return np.array(rows)


def _deduplicate_grouped(
    candidates: Sequence[Candidate],
    structures: Sequence[Atoms],
    model: Model,
    *,
    min_distance: float,
) -> CandidateSet:
    """De-duplicate at the window-group level, keeping surviving windows whole."""
    groups: dict[str, list[int]] = {}
    for index, candidate in enumerate(candidates):
        groups.setdefault(candidate.window_group or f"_single:{index}", []).append(index)

    centres: dict[str, np.ndarray] = {}
    for name, members in groups.items():
        middle = members[len(members) // 2]
        feature = cached_descriptors([structures[middle]], model,
                                     tag=f"failure_export_group_{name}")[0]
        centres[name] = feature / (np.linalg.norm(feature) + 1e-12)

    kept: list[str] = []
    assignment: dict[str, str] = {}
    for name in groups:
        if not kept:
            kept.append(name)
            assignment[name] = name
            continue
        distance = np.array([np.linalg.norm(centres[k] - centres[name]) for k in kept])
        nearest = int(np.argmin(distance))
        if distance[nearest] < min_distance:
            assignment[name] = kept[nearest]
        else:
            assignment[name] = name
            kept.append(name)

    out_candidates: list[Candidate] = []
    out_structures: list[Atoms] = []
    for cluster_id, name in enumerate(kept):
        merged = [n for n, target in assignment.items() if target == name]
        others = [centres[k] for k in kept if k != name]
        novelty = float(min(np.linalg.norm(o - centres[name]) for o in others)) if others else float("nan")
        for index in groups[name]:
            candidate = candidates[index]
            candidate.cluster_id = cluster_id
            candidate.cluster_size = len(groups[name])
            candidate.novelty = novelty
            candidate.selection_rationale += (
                f" [window '{name}' kept whole ({len(groups[name])} frames); "
                f"merged duplicate windows: {[n for n in merged if n != name] or 'none'}]"
            )
            out_candidates.append(candidate)
            out_structures.append(structures[index])

    return CandidateSet(
        candidates=out_candidates,
        structures=out_structures,
        n_before_dedup=len(candidates),
        dedup_method=(
            f"window-level de-duplication on MACE descriptors (min distance {min_distance}); "
            f"{len(groups)} windows -> {len(kept)} kept, each retained in full"
        ),
    )


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------
def export_candidates(candidate_set: CandidateSet, directory: str | Path, *, prefix: str = "candidates") -> dict[str, Path]:
    """Write ``<prefix>.xyz`` and ``<prefix>_manifest.json``."""
    from ase.io import write as ase_write

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    xyz = directory / f"{prefix}.xyz"
    manifest = directory / f"{prefix}_manifest.json"

    if candidate_set.structures:
        tagged = []
        for candidate, atoms in zip(candidate_set.candidates, candidate_set.structures):
            clone = atoms.copy()
            clone.info.update(
                {
                    "candidate_id": candidate.candidate_id,
                    "source_metric": candidate.source_metric,
                    "level": candidate.level,
                    "suggested_protocol": candidate.suggested_protocol,
                    "cluster_id": candidate.cluster_id,
                }
            )
            tagged.append(clone)
        ase_write(str(xyz), tagged, format="extxyz")

    manifest.write_text(
        json.dumps(candidate_set.manifest(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {"structures": xyz, "manifest": manifest}
