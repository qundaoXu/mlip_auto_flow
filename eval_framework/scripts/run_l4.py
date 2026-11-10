#!/usr/bin/env python
"""Driver: L4 dynamics at the equilibrium liquid state points.

    python run_l4.py [--state all|canon_liquid_3000K|canon_liquid_1200K|canon_liquid_950K] [--window 0.2 0.8]

Reports the diffusion coefficient together with its fit window and the log-log
slope inside that window, and refuses to treat a ramping trajectory as an
equilibrium diffusion reference.

``--state all`` runs every active state point marked for L4: the 3000 K and
1200 K liquids and the 950 K supercooled liquid, all at 0.0309 at/A^3, so the
three are an Arrhenius series at one density.  Each is an ensemble over the
three AIMD replicas (:mod:`mlipeval.ensemble`).
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
from mlipeval.l4_dynamics import analyse_dynamics
from mlipeval.state_points import (
    STATES, get_state, missing_seeds, model_production, reference_production,
    reference_trajectory, states_used_by,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", default="all",
                    choices=[*(k for k, v in STATES.items() if "L4" in v.used_by), "all"])
    ap.add_argument("--window", type=float, nargs=2, default=(0.2, 0.8), metavar=("LO", "HI"))
    ap.add_argument("--check-ramp", action="store_true", help="also show the guard rejecting the quench")
    args = ap.parse_args()

    names = states_used_by("L4") if args.state == "all" else [args.state]
    for name in names:
        analyse_state(name, tuple(args.window))

    if args.check_ramp:
        print("\nguard check on the quench ramp (must be rejected):")
        ramp = reference_trajectory("canon_quench", max_frames=400)
        from mlipeval.l4_dynamics import assert_equilibrium_segment

        guard = assert_equilibrium_segment(ramp)
        print(f"  {guard}")
        print(f"  reason: {guard.failure_reason}")
    return 0


def analyse_state(name: str, window: tuple[float, float]) -> None:
    state = get_state(name)
    if missing := missing_seeds(state):
        print(f"\n=== {name}: SKIPPED -- no MLIP MD for seed(s) {missing} ===")
        return
    n = state.n_replicas
    references = [reference_production(state, r) for r in range(1, n + 1)]
    models = [model_production(state, seed) for seed in range(n)]
    print(f"\n=== {name} ({n} replica{'s' if n > 1 else ''}) ===")
    for r, (ref, mod) in enumerate(zip(references, models), start=1):
        print(f"  replica {r}: {json.dumps(ref.describe())}")
        print(f"  seed {r - 1}   : {json.dumps(mod.describe())}")

    first_pair = {}

    def pair(ref, mod):
        report = analyse_dynamics(ref, model_traj=mod, window_fraction=window,
                                  state_label=name, n_independent_references=n)
        if ref is references[0] and mod is models[0]:
            first_pair["report"] = report
        return report.results

    paired, null = run_ensemble(pair, references, models)
    results = combine(paired, null, n_references=n)
    out = results.save(paths.md_result_path(f"L4_{name}.json"))
    report = first_pair["report"]  # replica 1 / seed 0: the arrays the notebook plots

    np.savez(
        paths.data_path(f"l4_{name}.npz"),
        time_ref=report.msd_ref.time_fs,
        **{f"msd_ref_{k}": v for k, v in report.msd_ref.msd.items()},
        time_model=report.msd_model.time_fs if report.msd_model else np.zeros(0),
        **{f"msd_model_{k}": v for k, v in (report.msd_model.msd if report.msd_model else {}).items()},
        n_origins_ref=report.msd_ref.n_origins,
    )

    print(f"\nwrote {out}\n")
    print(results.to_frame().to_string(index=False, max_colwidth=48))
    print("\ndiffusion fits (replica 1 / seed 0)")
    for species, fit in report.fits_ref.items():
        model_fit = report.fits_model.get(species)
        print(f"  {species:4s} AIMD D={fit.d_cm2_per_s:.3e} cm^2/s  slope={fit.loglog_slope:.2f}  "
              f"window={fit.fit_window_fs[0]:.0f}-{fit.fit_window_fs[1]:.0f} fs" +
              (f"   MLIP D={model_fit.d_cm2_per_s:.3e}  slope={model_fit.loglog_slope:.2f}" if model_fit else ""))


if __name__ == "__main__":
    raise SystemExit(main())
