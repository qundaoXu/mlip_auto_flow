#!/usr/bin/env python
"""Driver: L5 pathway comparison along a temperature programme.

    python run_l5.py [--state all|canon_quench|canon_heating] [--samples 40]

Compares the whole temperature programme rather than only its end state, and
reports where each observable first leaves the reference band.  Two ramps:
the 1200 -> 300 K quench (15 K/ps) and the 300 -> 1200 K heating of the
quenched glass (~36 K/ps).

Each ramp has three independent AIMD replicas, so the comparison is an
ensemble (:mod:`mlipeval.ensemble`): the end-state errors are graded against
the replica-to-replica spread, and a first divergence that independent AIMD
chains also show is reported as not attributable to the model.
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
from mlipeval.l5_pathway import compare_pathway
from mlipeval.state_points import (
    CUTOFF_STATE, frozen_params, get_state, missing_seeds, model_production,
    ramp_states, reference_production,
)


def analyse_state(name: str, args) -> None:
    state = get_state(name)
    if missing := missing_seeds(state):
        print(f"\n=== {name}: SKIPPED -- no MLIP MD for seed(s) {missing} ===")
        return
    n = state.n_replicas
    cutoff_state = args.cutoff_state or CUTOFF_STATE
    print(f"\n=== {name} ({n} replicas) ===")
    references = [reference_production(state, r) for r in range(1, n + 1)]
    models = [model_production(state, seed) for seed in range(n)]
    for r, (ref, mod) in enumerate(zip(references, models), start=1):
        print(f"replica {r}: {json.dumps(ref.describe())}")
        print(f"seed {r - 1}   : {json.dumps(mod.describe())}")

    # A ramp has no single well-defined RDF minimum, so the cutoffs are taken
    # from the isothermal glass and frozen for the whole path -- stated
    # explicitly rather than re-derived frame by frame.
    params, _ = frozen_params(cutoff_state)
    print(f"frozen cutoffs from '{cutoff_state}': "
          + ", ".join(f"{k}={v:.2f}" for k, v in sorted(params.bond_cutoffs.items())))

    first_pair = {}

    def pair(ref, mod):
        pair_report = compare_pathway(
            ref,
            params,
            model_traj=mod,
            n_samples=args.samples,
            average_window=args.average_window,
            sigma_factor=args.sigma_factor,
            segments=state.programme,
            state_label=name,
            n_independent_references=n,
        )
        if ref is references[0] and mod is models[0]:
            first_pair["report"] = pair_report
        return pair_report.results

    paired, null = run_ensemble(pair, references, models)
    results = combine(paired, null, n_references=n)
    results.context["cutoff_provenance"] = params.provenance
    results.context["cutoff_source_state"] = cutoff_state
    out = results.save(paths.md_result_path(f"L5_{name}.json"))
    report = first_pair["report"]  # replica 1 / seed 0: the traces the notebook plots

    ref_d, model_d = report.trace_ref.as_dict(), report.trace_model.as_dict()
    np.savez(
        paths.data_path(f"l5_{name}.npz"),
        **{f"ref_{k}": v for k, v in ref_d.items()},
        **{f"model_{k}": v for k, v in model_d.items()},
        **{f"sigma_{k}": v for k, v in report.trace_ref.std.items()},
        **{f"band_{k}": v for k, v in report.bands.items()},
        divergences=json.dumps(report.divergences),
        average_window=report.trace_ref.average_window,
    )

    print(f"\nwrote {out}\n")
    print(results.to_frame().to_string(index=False, max_colwidth=52))
    print("\nfirst divergence along the programme, replica 1 / seed 0 "
          f"(band = {args.sigma_factor:g} x reference within-window scatter)")
    for key, div in report.divergences.items():
        where = "never" if div["progress"] is None else f"{div['progress']:.2f} of the programme"
        band = float(np.nanmean(report.bands[key]))
        print(f"  {key:26s} {where:28s} band~{band:.4f}  max|delta| = {div['max_delta']:.4f} "
              f"({div['max_normalised']:.2f} sigma)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--average-window", type=int, default=8,
                    help="frames averaged per sample point; also sets the reference scatter estimate")
    ap.add_argument("--sigma-factor", type=float, default=3.0,
                    help="tolerance band = sigma_factor x the reference's own within-window scatter")
    ap.add_argument("--state", default="all", choices=[*ramp_states(), "all"])
    ap.add_argument("--cutoff-state", default=None,
                    help="state point whose frozen cutoffs are reused (a ramp has no single RDF minimum); "
                         f"default {CUTOFF_STATE}")
    args = ap.parse_args()

    for name in ramp_states() if args.state == "all" else [args.state]:
        analyse_state(name, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
