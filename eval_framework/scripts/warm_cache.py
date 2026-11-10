#!/usr/bin/env python
"""Populate every cache the notebooks read, so they never compute.

    python warm_cache.py

The delivered notebooks are meant to be opened and re-run interactively -- often
on the login node, where there is no GPU and where heavy work is not allowed.
They therefore read precomputed artifacts only: ``model_spec_only()`` gives them
the pinned identity without loading the network, and every model-derived array
comes from ``eval/data/cache``.

This driver is what fills that cache, from inside a PBS job:

* the six binary potential energy curves (~2100 dimer evaluations, nb02);
* MACE descriptors for the L0 worst-frame candidates of the diagnostic set (nb01);
* MACE descriptors for the L5 divergence windows of the canonical quench (nb06).

If a notebook is opened before this has run, the cache helpers raise with a
message pointing back here rather than silently evaluating MACE on the head node.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mlipeval import datasets, paths
from mlipeval.cache import cached_binary_pecs, cached_descriptors, cached_predict
from mlipeval.calculators import build_model
from mlipeval.failure_export import collect_from_l0, collect_from_l5, deduplicate
from mlipeval.l0_reference import evaluate_l0
from mlipeval.l5_pathway import compare_pathway
from mlipeval.pbs import describe_environment, require_compute_node
from mlipeval.state_points import CUTOFF_STATE, frozen_params, get_state, production_segments

SPECIES = ("Ge", "Sb", "Te")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--allow-login-node", action="store_true")
    args = ap.parse_args()
    require_compute_node("cache warming (PECs and MACE descriptors)",
                         allow_login_node=args.allow_login_node)

    import json

    print(json.dumps(describe_environment(), indent=2))
    model = build_model(threads=args.threads, device=args.device)
    print(f"model: {model.spec.name} on {model.spec.device}")

    print("\n[1/3] binary potential energy curves (nb02)")
    pecs = cached_binary_pecs(model, SPECIES, r_min=1.0, r_max=8.0, dr=0.02, refresh=args.refresh)
    print(f"      {len(pecs)} curves: " + ", ".join(sorted(pecs)))

    print("\n[2/3] L0 worst-frame descriptors (nb01)")
    labelled = datasets.labelled_diagnostic()
    predictions = cached_predict(labelled.frames, model, tag=f"L0_diagnostic_{len(labelled)}")
    report = evaluate_l0(labelled, model, predictions=predictions,
                         tags={"subset": labelled.tags["subset"]})
    candidates, structures = collect_from_l0(report, labelled=labelled, n_frames=10)
    cached_descriptors(structures, model, tag="failure_export", refresh=args.refresh)
    kept = deduplicate(candidates, structures, model, min_distance=0.02)
    print(f"      {len(structures)} structures -> {len(kept)} kept ({kept.dedup_method})")

    print("\n[3/3] L5 divergence-window descriptors (nb06)")
    quench = get_state("canon_quench")
    segments = production_segments(quench)
    params, _ = frozen_params(CUTOFF_STATE)
    pathway = compare_pathway(segments["reference"], params, model_traj=segments["model"],
                              n_samples=40, average_window=8, sigma_factor=3.0,
                              segments=quench.programme, state_label=quench.name)
    l5_candidates, l5_structures = collect_from_l5(pathway, segments["reference"], window=3)
    if l5_structures:
        kept5 = deduplicate(l5_candidates, l5_structures, model, min_distance=0.02)
        print(f"      {len(l5_structures)} frames -> {len(kept5)} kept ({kept5.dedup_method})")
    else:
        print("      no divergence windows to cache")

    cache_dir = paths.DATA / "cache"
    total = sum(f.stat().st_size for f in cache_dir.glob("*.npz"))
    print(f"\ncache: {len(list(cache_dir.glob('*.npz')))} files, {total/1e6:.1f} MB in {cache_dir}")
    print("notebooks can now be opened on the login node without computing anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
