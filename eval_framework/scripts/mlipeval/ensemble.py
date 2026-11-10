"""Replica ensembles: grade a model against independent AIMD replicas.

With one AIMD chain per state point ``sigma_ref`` could not be measured and
every L3-L5 number had to stay ``provisional``.  The phase-1 canonical state
points have three independent replicas, and this module turns that into a
grade without changing any analysis function:

* **paired** -- the existing analysis run on (AIMD replica ``r``, MLIP seed
  ``r - 1``) for every replica.  The ensemble value of a metric is the mean of
  the paired values; the per-pair values travel in ``extra.per_replica``.
* **null** -- the *same* analysis run on (replica ``i``, replica ``j``) for every
  pair ``i < j``: what the metric reads when the "model" is just another AIMD
  chain.  ``sigma_ref`` is the root-mean-square of those null values.

So ``z_eff = |paired mean| / sigma_ref <= 1`` (grade A) says the model is no
further from AIMD than one AIMD replica is from another -- the framework's
``sigma_eff = max(sigma_ref, d_protocol/2)`` with ``d_protocol`` absent, because
the reference and the comparison are both AIMD under one protocol.

Only *distances* are graded: metrics that carry a signed ``extra.difference``
(model minus reference) or whose value is itself a distance or an error.
Everything else -- raw model values such as ``L4.diffusion.X.D``, the
first-divergence position, reference-only and self-consistency checks -- is
averaged and reported with its per-pair values, but never given a grade.
"""

from __future__ import annotations

import copy
import itertools
import math
import re
from typing import Any, Callable, Sequence

from .schema import MetricResult, ResultSet, Status, grade_metric

#: Metric ids whose value is itself a distance or a signed error.
DISTANCE_METRICS = re.compile(r"(wasserstein1|jensen_shannon|delta_log10D|final_state_error)@")

SIGMA_REF_DEFINITION = (
    "sigma_ref = RMS of the same metric over all pairs of independent AIMD replicas "
    "(the value the metric takes when the 'model' is another AIMD chain); "
    "z_eff = |mean over model-vs-replica pairs| / sigma_ref"
)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def _graded_quantity(result: MetricResult | None) -> float | None:
    """The signed distance a metric reports, or ``None`` if it is not a distance."""
    if result is None:
        return None
    difference = _number((result.extra or {}).get("difference"))
    if difference is not None:
        return difference
    if DISTANCE_METRICS.search(result.metric_id):
        return _number(result.value)
    return None


def _mean(values: Sequence[float | None]) -> float | None:
    numbers = [v for v in values if v is not None]
    return sum(numbers) / len(numbers) if numbers else None


def replica_pairs(n_replicas: int) -> list[tuple[int, int]]:
    """Null pairs ``(i, j)``, ``i < j``, over replicas numbered from 1."""
    return list(itertools.combinations(range(1, n_replicas + 1), 2))


def run_ensemble(
    analyse: Callable[[Any, Any], ResultSet],
    references: Sequence[Any],
    models: Sequence[Any],
) -> tuple[list[ResultSet], list[ResultSet]]:
    """Run one analysis over every model-vs-replica pair and every replica pair.

    ``references[r - 1]`` is replica ``r``; ``models[r - 1]`` is the MLIP seed
    that mirrors it.  ``analyse(reference, model)`` returns a ResultSet.
    """
    paired = [analyse(ref, mod) for ref, mod in zip(references, models)]
    null = [analyse(references[i - 1], references[j - 1]) for i, j in replica_pairs(len(references))]
    return paired, null


def combine(
    paired: Sequence[ResultSet],
    null: Sequence[ResultSet] = (),
    *,
    n_references: int,
) -> ResultSet:
    """One ResultSet whose metrics are the ensemble over the paired runs.

    The metric ids, ``value`` and ``extra.difference`` keep the shapes the
    single-pair analysis produced, so :mod:`mlipeval.ranking` reads the result
    unchanged; with ``n_references < 2`` (a legacy state point) nothing is
    graded and every comparison stays provisional, exactly as before.
    """
    first = paired[0]
    out = ResultSet(
        first.level,
        model=first.model,
        **{
            **first.context,
            "n_independent_references": n_references,
            "replica_pairing": [f"replica {r} <-> seed {r - 1}" for r in range(1, len(paired) + 1)],
            "null_pairs": [f"replica {i} vs replica {j}" for i, j in replica_pairs(n_references)][: len(null)],
            "grading": SIGMA_REF_DEFINITION if null else "single reference: no sigma_ref, nothing graded",
        },
    )
    for template in first:
        mid = template.metric_id
        group = [rs.get(mid) for rs in paired]
        merged = copy.deepcopy(template)
        merged.extra = dict(template.extra or {})

        values = [_number(r.value) if r is not None else None for r in group]
        if any(v is not None for v in values):
            merged.value = _mean(values)
        for key in ("reference", "model", "difference"):
            parts = [_number((r.extra or {}).get(key)) if r is not None else None for r in group]
            if all(p is not None for p in parts):
                merged.extra[key] = _mean(parts)
        merged.extra["per_replica"] = [
            None if r is None else {
                "replica": i + 1,
                "seed": i,
                "value": r.value,
                "difference": (r.extra or {}).get("difference"),
                "status": r.status.value,
                "failure_reason": r.failure_reason,
            }
            for i, r in enumerate(group)
        ]

        failed = [r for r in group if r is not None and r.status is Status.FAILED]
        if failed:
            null_group = [rs.get(mid) for rs in null]
            null_failed = sum(1 for r in null_group if r is not None and r.status is Status.FAILED)
            merged.extra["null_failed"] = f"{null_failed}/{len(null_group)}"
            if null_group and null_failed / len(null_group) >= len(failed) / len(group):
                # Independent AIMD chains fail this check as often as the model
                # does: it is a property of the state point, not a model error.
                merged.status = Status.PROVISIONAL
                merged.failure_reason = (
                    f"{len(failed)}/{len(group)} model pairs fail, but so do {null_failed}/{len(null_group)} "
                    f"AIMD replica pairs: not attributable to the model ({failed[0].failure_reason})"
                )
            else:
                merged.status = Status.FAILED
                merged.failure_reason = f"{len(failed)}/{len(group)} replica pairs: {failed[0].failure_reason}"
            out.add(merged)
            continue
        if any(r is None or r.status is Status.PROVISIONAL for r in group):
            merged.status = Status.PROVISIONAL

        quantity = _graded_quantity(merged)
        if merged.status is Status.PROVISIONAL and quantity is not None and len(paired) == n_references:
            null_values = [_graded_quantity(rs.get(mid)) for rs in null]
            null_values = [v for v in null_values if v is not None]
            sigma_ref = (
                math.sqrt(sum(v * v for v in null_values) / len(null_values)) if null_values else None
            )
            merged.extra["null_per_pair"] = null_values
            merged.extra["sigma_ref"] = sigma_ref
            merged.extra.pop("grading_note", None)
            grade_metric(
                merged,
                quantity,
                sigma_ref=sigma_ref,
                n_independent_references=n_references if null_values else 1,
            )
        out.add(merged)
    return out
