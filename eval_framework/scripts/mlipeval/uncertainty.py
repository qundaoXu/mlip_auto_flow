"""Statistical uncertainty for trajectory-averaged observables.

L3 produces well over a hundred numbers per state point, and without an error
bar none of them can be acted on: there is no way to tell a real model error
from the scatter of a finite trajectory.  This module supplies the two error
scales that *can* be measured with the data actually available, and is explicit
that neither of them is ``sigma_ref``:

``block SEM``
    Split one trajectory into contiguous blocks, evaluate the observable on
    each, and take the standard error of the mean across blocks.  This is the
    **statistical error of the estimate within one chain**.  It answers "is this
    number converged?", not "how much do independent chains differ?".

``seed scatter``
    Evaluate the same observable on two MLIP runs that differ only in the random
    seed.  This is **chain-to-chain variability at fixed physics**, measured on
    the model side.  It answers "would an identical model reproduce this
    difference by chance?".

``sigma_ref`` in the framework's grading formula is chain-to-chain variability on
the *reference* side, which needs independent AIMD replicas that do not exist
yet.  :func:`compare_with_uncertainty` therefore never assigns a grade -- it
only labels each observable as resolved, within statistical noise, or within
seed scatter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from .io_vasp import Trajectory
from .neighbors import (
    StructureParams,
    all_pairs,
    coordination_numbers,
    first_peak,
    homopolar_fraction,
    pair_key,
    partial_rdf,
    tetrahedral_order,
)


@dataclass
class BlockEstimate:
    """Mean of an observable and the statistical error of that mean."""

    mean: float
    sem: float
    n_blocks: int
    block_values: np.ndarray

    def __str__(self) -> str:  # pragma: no cover - display helper
        return f"{self.mean:.4g} +- {self.sem:.2g}"

    def to_dict(self) -> dict[str, object]:
        return {
            "mean": self.mean,
            "sem": self.sem,
            "n_blocks": self.n_blocks,
            "block_values": self.block_values.tolist(),
        }


def _estimate(values: Sequence[float]) -> BlockEstimate:
    array = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if array.size == 0:
        return BlockEstimate(np.nan, np.nan, 0, array)
    if array.size == 1:
        return BlockEstimate(float(array[0]), np.nan, 1, array)
    return BlockEstimate(
        mean=float(array.mean()),
        sem=float(array.std(ddof=1) / np.sqrt(array.size)),
        n_blocks=int(array.size),
        block_values=array,
    )


# --------------------------------------------------------------------------
# the scalar observables L3 reports
# --------------------------------------------------------------------------
def l3_scalars(
    traj: Trajectory,
    params: StructureParams,
    *,
    stride: int = 2,
    ring_stride: int = 5,
    tetrahedral_species: str = "Ge",
    tetrahedral_threshold: float = 0.8,
    cache: bool = True,
) -> dict[str, float]:
    """Every scalar L3 reports, evaluated on one (sub)trajectory.

    Kept deliberately flat -- one dict of floats -- so the same function can be
    applied to a whole trajectory, to a block of it, or to a different seed, and
    the results compared key by key.
    """
    from .l3b_medium_range import altbc, king_rings

    out: dict[str, float] = {}
    symbols = np.asarray(traj.symbols)

    # coordination numbers
    totals = {s: [] for s in traj.species}
    totals["all"] = []
    for k in range(0, len(traj), stride):
        cn = coordination_numbers(traj.atoms(k), params)
        for species in traj.species:
            totals[species].append(float(cn[symbols == species].mean()))
        totals["all"].append(float(cn.mean()))
    for key, values in totals.items():
        out[f"cn_mean.{key}"] = float(np.mean(values)) if values else np.nan

    # homopolar bonds
    for key, value in homopolar_fraction(traj, params, stride=stride).items():
        out[f"homopolar.{key}"] = float(value)

    # tetrahedral order
    q = tetrahedral_order(traj, params, species=tetrahedral_species, stride=stride)
    out[f"tetrahedral_q.{tetrahedral_species}.mean"] = float(q.mean()) if q.size else np.nan
    out[f"tetrahedral_q.{tetrahedral_species}.fraction_gt_{tetrahedral_threshold:g}"] = (
        float((q > tetrahedral_threshold).mean()) if q.size else np.nan
    )

    # RDF peaks
    rdf = partial_rdf(traj, params, stride=stride)
    r = rdf["r"]
    for a, b in all_pairs(traj.species):
        key = pair_key(a, b)
        peak = first_peak(r, rdf[key])
        out[f"rdf.{key}.r_peak"] = peak["r_peak"]
        out[f"rdf.{key}.g_peak"] = peak["g_peak"]

    # rings
    sizes = np.arange(3, params.max_ring_size + 1)
    counts = np.zeros(sizes.size)
    n_frames = 0
    for k in range(0, len(traj), ring_stride):
        rings = king_rings(traj.atoms(k), params)
        for size, members in rings.items():
            if 3 <= size <= params.max_ring_size:
                counts[size - 3] += len(members)
        n_frames += 1
    counts /= max(n_frames, 1) * traj.n_atoms
    for size, value in zip(sizes, counts):
        out[f"rings.size{size}"] = float(value)

    # ALTBC asymmetry
    a = altbc(traj, params, stride=ring_stride)
    out["altbc.asymmetry"] = float(a["asymmetry"][0])
    return out


# --------------------------------------------------------------------------
# block resampling
# --------------------------------------------------------------------------
def block_estimates(
    traj: Trajectory,
    params: StructureParams,
    *,
    n_blocks: int = 5,
    observables: Callable[..., dict[str, float]] = l3_scalars,
    **kwargs,
) -> dict[str, BlockEstimate]:
    """Evaluate ``observables`` on ``n_blocks`` contiguous blocks of ``traj``.

    Contiguous blocks (not random subsamples) are used because MD frames are
    strongly autocorrelated: random subsampling would badly underestimate the
    error by treating correlated frames as independent.
    """
    if len(traj) < 2 * n_blocks:
        n_blocks = max(2, len(traj) // 2)
    edges = np.linspace(0, len(traj), n_blocks + 1).astype(int)

    per_block: list[dict[str, float]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi - lo < 2:
            continue
        per_block.append(observables(traj.slice(int(lo), int(hi)), params, **kwargs))

    keys = sorted({k for block in per_block for k in block})
    return {key: _estimate([block.get(key, np.nan) for block in per_block]) for key in keys}


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------
@dataclass
class ObservableComparison:
    """One observable, compared with every error scale that is measurable."""

    name: str
    reference: BlockEstimate
    model: BlockEstimate
    difference: float
    combined_sem: float
    z_statistical: float
    seed_scatter: float | None        # raw max-min across seeds
    effective_scatter: float | None   # floored by the combined SEM (see below)
    seed_ratio: float | None          # |difference| / effective scatter; the decision number
    n_seeds: int
    verdict: str

    def to_dict(self) -> dict[str, object]:
        return {
            "observable": self.name,
            "reference_mean": self.reference.mean,
            "reference_sem": self.reference.sem,
            "model_mean": self.model.mean,
            "model_sem": self.model.sem,
            "difference": self.difference,
            "combined_sem": self.combined_sem,
            "z_statistical": self.z_statistical,
            "seed_scatter": self.seed_scatter,
            "effective_scatter": self.effective_scatter,
            "seed_ratio": self.seed_ratio,
            "n_seeds": self.n_seeds,
            "verdict": self.verdict,
        }


#: What each verdict means, quoted into the result file so it travels with the number.
VERDICTS = {
    "resolved": (
        "|difference| exceeds both the combined block SEM (x z_threshold) and the "
        "model's own seed-to-seed scatter: a real difference at the available statistics"
    ),
    "within_statistical_noise": (
        "|difference| is not larger than the combined block SEM: the trajectories are "
        "too short to say the two sides differ at all"
    ),
    "within_seed_scatter": (
        "|difference| is statistically significant but no larger than what the same model "
        "produces by changing only its random seed (scatter floored by the combined SEM): "
        "not attributable to model error"
    ),
    "unavailable": "one of the two estimates is missing or non-finite",
}


def compare_with_uncertainty(
    reference: Mapping[str, BlockEstimate],
    model: Mapping[str, BlockEstimate],
    *,
    seed_scatter: Mapping[str, float] | None = None,
    n_seeds: int = 1,
    z_threshold: float = 2.0,
) -> list[ObservableComparison]:
    """Compare two block-estimated observable sets with explicit error scales.

    ``n_seeds`` is recorded on every comparison: a scatter measured from two
    seeds is itself a very noisy estimate, and the floor applied below is what
    keeps that noise from manufacturing spurious significance.
    """
    out: list[ObservableComparison] = []
    for name in sorted(set(reference) & set(model)):
        ref, mod = reference[name], model[name]
        if not (np.isfinite(ref.mean) and np.isfinite(mod.mean)):
            out.append(ObservableComparison(
                name, ref, mod, np.nan, np.nan, np.nan, None, None, None, n_seeds, "unavailable"
            ))
            continue
        difference = mod.mean - ref.mean
        sem = float(np.sqrt(np.nan_to_num(ref.sem) ** 2 + np.nan_to_num(mod.sem) ** 2))
        z = float(abs(difference) / sem) if sem > 0 else float("inf")
        scatter = None if seed_scatter is None else seed_scatter.get(name)

        # Chain-to-chain scatter cannot meaningfully be smaller than the
        # statistical error of the estimates it is computed from.  With only two
        # seeds, max-min can land arbitrarily close to zero by chance and would
        # otherwise produce an absurd significance ratio, so it is floored.
        effective = None
        if scatter is not None and np.isfinite(scatter):
            effective = float(max(abs(scatter), sem)) if sem > 0 else float(abs(scatter))
        ratio = (
            float(abs(difference) / effective)
            if effective is not None and effective > 0
            else None
        )
        if z <= z_threshold:
            verdict = "within_statistical_noise"
        elif effective is not None and abs(difference) <= effective:
            verdict = "within_seed_scatter"
        else:
            verdict = "resolved"
        out.append(ObservableComparison(
            name, ref, mod, difference, sem, z, scatter, effective, ratio, n_seeds, verdict
        ))
    return out


def seed_scatter_from(estimates: Sequence[Mapping[str, BlockEstimate]]) -> dict[str, float]:
    """Spread of an observable across MLIP runs that differ only in the seed."""
    if len(estimates) < 2:
        return {}
    keys = sorted(set.intersection(*[set(e) for e in estimates]))
    out: dict[str, float] = {}
    for key in keys:
        means = np.array([e[key].mean for e in estimates], dtype=float)
        means = means[np.isfinite(means)]
        out[key] = float(means.max() - means.min()) if means.size > 1 else np.nan
    return out
