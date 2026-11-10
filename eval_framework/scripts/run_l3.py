#!/usr/bin/env python
"""Driver: L3a short-range order and L3b medium-range order.

    python run_l3.py [--state canon_amorphous_300K|...|all] [--cutoffs-from STATE]

The bond cutoffs, bin widths and ring algorithm are derived once from the AIMD
reference of the chosen state point and then applied unchanged to the MLIP
trajectory.  Both sides of the comparison therefore share every post-processing
knob, which is the L3 precondition in ``validation_methods_L0-L5.md``.

For a canonical state point (three AIMD replicas) the battery runs on every
model-vs-replica pair and every replica-vs-replica pair, and
:func:`mlipeval.ensemble.combine` grades the ensemble against the replica spread.
The cutoffs are frozen once from the *pooled* replicas, so all six comparisons
share one parameter set.  Frame strides come from the state point
(``l3_stride``/``ring_stride``) unless overridden.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from mlipeval import paths
from mlipeval.ensemble import combine, run_ensemble
from mlipeval.l3a_short_range import analyse_short_range
from mlipeval.l3b_medium_range import analyse_medium_range
from mlipeval.state_points import (
    STATES, frozen_params, get_state, isothermal_states, missing_seeds,
    model_production, reference_production,
)


def analyse_state(
    name: str,
    *,
    ref_stride: int | None,
    model_stride: int | None,
    ring_stride: int | None,
    cutoffs_from: str | None = None,
) -> None:
    state = get_state(name)
    if missing := missing_seeds(state):
        print(f"\n=== {name}: SKIPPED -- no MLIP MD for seed(s) {missing}; "
              f"run `run_mlip_md.py --state {name} --seed K` first ===")
        return
    # Both sides are sampled at the same interval, so one stride serves both.
    ref_stride = ref_stride or state.l3_stride
    model_stride = model_stride or state.l3_stride
    ring_stride = ring_stride or state.ring_stride
    ring_model_stride = ring_stride
    n = state.n_replicas
    references = [reference_production(state, r) for r in range(1, n + 1)]
    models = [model_production(state, seed) for seed in range(n)]
    reference, model = references[0], models[0]
    print(f"\n=== {name} ({n} replica{'s' if n > 1 else ''}) ===")
    for r, (ref, mod) in enumerate(zip(references, models), start=1):
        print(f"  replica {r} : {json.dumps(ref.describe())}")
        print(f"  seed {r - 1}    : {json.dumps(mod.describe())}")

    cutoff_source = cutoffs_from or name
    params, rdf_seed = frozen_params(cutoff_source, stride=ref_stride)
    print(f"  frozen bond cutoffs (AIMD partial RDF first minima of '{cutoff_source}'):")
    n_fallback = 0
    for pair, info in params.cutoff_diagnostics.items():
        if pair == "_total":
            continue
        n_fallback += int(not info["well_defined"])
        print(
            f"    {pair:6s} cutoff={params.bond_cutoffs[pair]:.3f} A  depth={info['depth']:+.2f}  "
            f"{info.get('source', '')}"
        )
    n_pairs = len(params.bond_cutoffs)
    if n_fallback > n_pairs // 2:
        print(
            f"  WARNING: {n_fallback}/{n_pairs} pairs have no well-defined RDF first minimum.\n"
            f"           At high temperature the shells merge and the per-state prescription\n"
            f"           degenerates: the fallback cutoff then counts the second shell and\n"
            f"           inflates CN. Re-run with --cutoffs-from <isothermal state> and say so."
        )

    # The replica-1 / seed-0 reports are kept: their arrays feed the notebooks.
    first_pair: dict[str, object] = {}

    def is_model(traj) -> bool:
        return any(traj is m for m in models)

    def l3a_pair(ref, mod):
        # replica-vs-replica pairs use the reference stride on both sides
        report = analyse_short_range(
            ref, params, model_traj=mod, stride=ref_stride,
            model_stride=model_stride if is_model(mod) else ref_stride,
            state_label=name, n_independent_references=n,
        )
        if ref is reference and mod is model:
            first_pair["L3a"] = report
        return report.results

    def l3b_pair(ref, mod):
        report = analyse_medium_range(
            ref, params, model_traj=mod, stride=ring_stride,
            model_stride=ring_model_stride if is_model(mod) else ring_stride,
            state_label=name, sq_stride=ring_stride, n_independent_references=n,
        )
        if ref is reference and mod is model:
            first_pair["L3b"] = report
        return report.results

    combined = {}
    for tag, pair_fn in (("L3a", l3a_pair), ("L3b", l3b_pair)):
        paired, null = run_ensemble(pair_fn, references, models)
        results = combine(paired, null, n_references=n)
        results.context["cutoff_source_state"] = cutoff_source
        results.save(paths.md_result_path(f"{tag}_{name}.json"))
        graded = sum(1 for r in results if r.z_eff is not None)
        print(f"  {tag}: {len(results)} metrics, {graded} graded -> results/{tag}_{name}.json")
        combined[tag] = results

    l3a, l3b = first_pair["L3a"], first_pair["L3b"]

    np.savez_compressed(
        paths.data_path(f"l3_{name}.npz"),
        r=l3a.rdf_ref["r"],
        **{f"rdf_ref_{k}": v for k, v in l3a.rdf_ref.items() if k not in ("r", "_meta")},
        **{f"rdf_model_{k}": v for k, v in (l3a.rdf_model or {}).items() if k not in ("r", "_meta")},
        angle=l3a.adf_ref["angle"],
        **{f"adf_ref_{k}": v for k, v in l3a.adf_ref.items() if k != "angle"},
        **{f"adf_model_{k}": v for k, v in (l3a.adf_model or {}).items() if k != "angle"},
        cn_values=l3a.cn_ref["cn_values"],
        **{f"cn_ref_{k}": v for k, v in l3a.cn_ref.items() if k not in ("cn_values", "_meta")},
        **{f"cn_model_{k}": v for k, v in (l3a.cn_model or {}).items() if k not in ("cn_values", "_meta")},
        tetra_ref=l3a.tetra_ref,
        tetra_model=l3a.tetra_model if l3a.tetra_model is not None else np.zeros(0),
        ring_sizes=l3b.rings_ref["sizes"],
        rings_ref=l3b.rings_ref["rings_per_atom"],
        rings_model=l3b.rings_model["rings_per_atom"] if l3b.rings_model else np.zeros(0),
        planarity_ref=l3b.rings_ref["planarity_4ring"],
        planarity_model=l3b.rings_model["planarity_4ring"] if l3b.rings_model else np.zeros(0),
        altbc_r=l3b.altbc_ref["r"],
        altbc_ref=l3b.altbc_ref["hist"],
        altbc_model=l3b.altbc_model["hist"] if l3b.altbc_model else np.zeros(0),
        q=l3b.sq_ref["q"],
        sq_ref=l3b.sq_ref["S"],
        sq_model=l3b.sq_model["S"] if l3b.sq_model else np.zeros(0),
        bond_cutoffs=json.dumps(params.bond_cutoffs),
        cutoff_diagnostics=json.dumps(params.cutoff_diagnostics),
        provenance=params.provenance,
        cutoff_source=cutoff_source,
    )

    for tag, results in combined.items():
        frame = results.to_frame()
        interesting = frame[frame.metric_id.str.contains("cn\\.|homopolar|rings|fsdp|altbc|tetra")]
        if not interesting.empty:
            print(f"  --- {tag} highlights ---")
            print(interesting.to_string(index=False, max_colwidth=48))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", default="all",
                    choices=[*(k for k, v in STATES.items() if not v.is_ramp), "all"])
    ap.add_argument("--stride", type=int, default=None,
                    help="frame stride for RDF/ADF/CN on the reference (default: the state point's l3_stride)")
    ap.add_argument("--model-stride", type=int, default=None)
    ap.add_argument("--ring-stride", type=int, default=None,
                    help="rings and ALTBC are the expensive part (default: the state point's ring_stride)")
    ap.add_argument(
        "--cutoffs-from",
        default=None,
        metavar="STATE",
        help=(
            "inherit the frozen bond cutoffs from another state point instead of deriving them "
            "in place; needed for a hot liquid, where the partial RDF minima wash out"
        ),
    )
    args = ap.parse_args()

    names = isothermal_states() if args.state == "all" else [args.state]
    for name in names:
        analyse_state(name, ref_stride=args.stride, model_stride=args.model_stride,
                      ring_stride=args.ring_stride, cutoffs_from=args.cutoffs_from)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
