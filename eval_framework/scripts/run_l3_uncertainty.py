#!/usr/bin/env python
"""Driver: attach measurable error bars to the L3 observables.

    python run_l3_uncertainty.py [--state canon_amorphous_300K] [--blocks 5]

L3 emits well over a hundred numbers per state point.  Without an error scale
none of them is actionable, so this driver reports, for every scalar L3
observable:

* the **block SEM** of the AIMD estimate and of the MLIP estimate
  (statistical error within one chain);
* the **seed scatter** of the model, when a second MLIP seed exists
  (chain-to-chain variability at fixed physics);
* a verdict: ``resolved`` / ``within_statistical_noise`` / ``within_seed_scatter``.

Neither error scale is ``sigma_ref`` (chain-to-chain variation on the
*reference* side), so no grade is assigned here -- only a statement of what the
data can and cannot distinguish.  For the canonical state points, which do have
independent AIMD replicas, the grades are in the ``run_l3.py`` results
(:mod:`mlipeval.ensemble`); this driver looks at the replica-1 / seed-0 pair and
the model's seed scatter.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from mlipeval import paths
from mlipeval.schema import EvidenceTier, MetricResult, ResultSet, Status
from mlipeval.state_points import (
    STATES, frozen_params, get_state, missing_seeds, mlip_trajectory, production_segments,
)
from mlipeval.uncertainty import (
    VERDICTS, block_estimates, compare_with_uncertainty, seed_scatter_from,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", default="canon_amorphous_300K", choices=[k for k, v in STATES.items() if not v.is_ramp])
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--ring-stride", type=int, default=5)
    ap.add_argument("--z-threshold", type=float, default=2.0)
    ap.add_argument("--cutoffs-from", default=None, metavar="STATE")
    args = ap.parse_args()

    state = get_state(args.state)
    if missing := missing_seeds(state):
        print(f"=== {args.state}: SKIPPED -- no MLIP MD for seed(s) {missing} ===")
        return 0
    if state.is_ramp:
        print(f"'{args.state}' 是一条升降温程序，把它做时间平均没有物理意义。"
              "路径级的比较见 run_l5.py。")
        return 1
    segments = production_segments(state)
    reference, model = segments["reference"], segments["model"]

    cutoff_source = args.cutoffs_from or args.state
    params, _ = frozen_params(cutoff_source)
    n_fallback = sum(
        1 for k, v in params.cutoff_diagnostics.items() if k != "_total" and not v["well_defined"]
    )
    if n_fallback > len(params.bond_cutoffs) // 2:
        print(f"WARNING: {n_fallback}/{len(params.bond_cutoffs)} 对没有良定义的 RDF 第一极小；"
              "高温下该处方会退化，回退截断会数到第二壳层。改用 --cutoffs-from <等温状态点>。\n")
    print(f"state          : {args.state}")
    print(f"cutoff 来源    : {cutoff_source}")
    print(f"blocks         : {args.blocks} (contiguous)")
    print(f"reference      : {reference.name}, {len(reference)} frames")
    print(f"model (seed 0) : {model.name}, {len(model)} frames")

    kwargs = dict(stride=args.stride, ring_stride=args.ring_stride)
    print("\n分块估计 AIMD ...", flush=True)
    ref_est = block_estimates(reference, params, n_blocks=args.blocks, **kwargs)
    print("分块估计 MLIP seed 0 ...", flush=True)
    mod_est = block_estimates(model, params, n_blocks=args.blocks, **kwargs)

    seeds = state.available_seeds()
    scatter = None
    other_est = None
    if len(seeds) > 1:
        other = mlip_trajectory(state, seed=seeds[1])
        keep = 1.0 - state.equilibration_fraction
        other = other if state.is_ramp else other.last_fraction(keep)
        print(f"分块估计 MLIP seed {seeds[1]} ...", flush=True)
        other_est = block_estimates(other, params, n_blocks=args.blocks, **kwargs)
        scatter = seed_scatter_from([mod_est, other_est])
    else:
        print(f"\n只有 seed 0。跑 `python run_mlip_md.py --state {args.state} --seed 1` "
              "后重跑本脚本即可加入种子散布。")

    comparisons = compare_with_uncertainty(ref_est, mod_est, seed_scatter=scatter,
                                           n_seeds=len(seeds), z_threshold=args.z_threshold)

    results = ResultSet(
        "L3-uncertainty",
        state=args.state,
        cutoff_source_state=cutoff_source,
        frozen_params=params.to_dict(),
        n_blocks=args.blocks,
        z_threshold=args.z_threshold,
        seeds_available=seeds,
        verdict_definitions=VERDICTS,
        note=(
            "block SEM = statistical error within one chain; seed scatter = chain-to-chain "
            "variability of the model. Neither is sigma_ref (which needs independent AIMD "
            "replicas), so no grade is assigned."
        ),
    )
    for c in comparisons:
        results.add(MetricResult(
            metric_id=f"L3u.{c.name}@{args.state}",
            level="L3-uncertainty",
            value=c.difference,
            unit="observable units (model - reference)",
            status=Status.PROVISIONAL,
            uncertainty=c.combined_sem,
            evidence_tier=EvidenceTier.A,
            reference_protocol=reference.protocol.protocol_id if reference.protocol else None,
            model_protocol=model.protocol.protocol_id if model.protocol else None,
            failure_reason=None if c.verdict == "resolved" else VERDICTS[c.verdict],
            extra=c.to_dict(),
        ))
    out = results.save(paths.md_result_path(f"L3_uncertainty_{args.state}.json"))

    import pandas as pd

    frame = pd.DataFrame([c.to_dict() for c in comparisons])
    print(f"\nwrote {out}\n")
    counts = frame.verdict.value_counts().to_dict()
    print("判定分布:", counts)
    print(f"  可分辨的观测量: {counts.get('resolved', 0)} / {len(frame)}")
    if frame["seed_ratio"].notna().any():
        top = frame[frame.verdict == "resolved"].nlargest(5, "seed_ratio")
        print(f"\n偏差最显著（|Δ| / 有效散布 最大；有效散布 = max(种子散布, 合并 SEM)，"
              f"种子数 = {len(seeds)}）:")
        for _, row in top.iterrows():
            print(f"  {row['observable']:34s} {row['reference_mean']:.4g} -> {row['model_mean']:.4g}"
                  f"   Δ={row['difference']:+.4g}   = {row['seed_ratio']:.1f}x 有效散布")

    for verdict in ("resolved", "within_seed_scatter", "within_statistical_noise"):
        sub = frame[frame.verdict == verdict]
        if sub.empty:
            continue
        print(f"\n--- {verdict} ({len(sub)}) ---")
        sort_key = "seed_ratio" if sub["seed_ratio"].notna().any() else "z_statistical"
        show = sub.sort_values(sort_key, ascending=False).head(20)
        print(show[["observable", "reference_mean", "model_mean", "difference",
                    "combined_sem", "z_statistical", "seed_scatter", "effective_scatter",
                    "seed_ratio"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
