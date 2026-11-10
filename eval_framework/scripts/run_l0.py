#!/usr/bin/env python
"""Driver: L0 numerical reference consistency.

    python run_l0.py [--set both|audit|diagnostic] [--limit N] [--refresh] [--threads T]

Two label sets, both ``gst225_pbe_canonical_v1`` and both phase-1 only, with
different jobs (``validation_methods_L0-L5.md`` section 3):

* **audit** -- the 120 H5 relabels (306-atom Ge68Sb68Te170 from the canonical
  AIMD chains: 950 K, heating, 3000 K melt, 1200 K, quench, 300 K glass).
  ``protected_audit``: it is what L0 is *scored* on, and nothing derived from it
  may steer new data.  Writes ``results/L0_audit.json``; failure export is never
  run on it.
* **diagnostic** -- every ionic step of the crystalline phase-1 tasks (L1 MP
  cells, L2 element references, H1 cubic seeds, H2 EOS).  ``diagnostic``: this
  is where worst frames are exported as candidates for the next DFT round.
  Frames the model was trained on are dropped first.  Writes
  ``results/L0_diagnostic.json`` and ``results/candidates/L0_diagnostic_*``.

Model predictions are cached under ``eval/data/cache``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mlipeval import datasets, models, paths
from mlipeval.cache import cached_predict
from mlipeval.calculators import build_model, stdout_progress
from mlipeval.failure_export import collect_from_l0, deduplicate, export_candidates
from mlipeval.l0_reference import evaluate_l0
from mlipeval.pbs import require_compute_node


def _predict(labelled, model, *, name: str, refresh: bool):
    predictions = cached_predict(
        labelled.frames, model, tag=f"L0_{name}_{len(labelled)}", refresh=refresh,
        progress=stdout_progress("  model "),
    )
    print(f"predictions   : {len(predictions)} frames in {predictions.wall_time_s:.1f}s "
          f"({len(predictions.failures)} failures)")
    return predictions


def run_audit(model, args) -> None:
    labelled = datasets.labelled_h5()
    if args.limit:
        labelled = labelled.subset([i < args.limit for i in range(len(labelled))])
    screen = json.loads(paths.H5_LEAKAGE.read_text())
    print(f"\n=== audit: {len(labelled)} H5 frames, role={labelled.role.value}, "
          f"leakage screen '{screen['verdict']}' ({screen['screened_at']}) ===")
    predictions = _predict(labelled, model, name="h5", refresh=args.refresh)
    report = evaluate_l0(
        labelled, model, predictions=predictions,
        tags={k: labelled.tags[k] for k in ("pool", "state_kind")},
        leakage_note=(f"H5 screened '{screen['verdict']}' against every GST training set in the "
                      f"workspace (sha256 {screen['sha256'][:12]}, threshold {screen['threshold_rdf']})"),
    )
    report.results.context["failure_export"] = "never run on this set: it is protected_audit"
    out = report.results.save(paths.result_path("L0_audit.json"))
    print(f"wrote {out}")
    print(report.results.to_frame().to_string(index=False, max_colwidth=44))


def run_diagnostic(model, args) -> None:
    labelled = datasets.labelled_diagnostic()
    seen = datasets.diagnostic_training_overlap(labelled, models.current_key())
    if seen.any():
        dropped = sorted({c.split(":")[0] for c in labelled.config_types()[seen]})
        print(f"dropping {int(seen.sum())} frame(s) in this model's training set: {dropped}")
        labelled = labelled.subset(~seen)
    if args.limit:
        labelled = labelled.subset([i < args.limit for i in range(len(labelled))])
    print(f"\n=== diagnostic: {len(labelled)} frames, role={labelled.role.value} ===")
    predictions = _predict(labelled, model, name="diagnostic", refresh=args.refresh)
    report = evaluate_l0(
        labelled, model, predictions=predictions,
        tags={"subset": labelled.tags["subset"]},
        leakage_note=(f"{int(seen.sum())} frame(s) inside this model's training set removed before "
                      "scoring (screen_diagnostic_leakage.py)"),
    )
    report.results.context["training_overlap_removed"] = int(seen.sum())

    candidates, structures = collect_from_l0(report, labelled=labelled, n_frames=args.candidates)
    try:
        kept = deduplicate(candidates, structures, model, min_distance=0.02)
        files = export_candidates(kept, paths.result_path("candidates"), prefix="L0_diagnostic")
        print(f"candidates    : {len(structures)} worst frames -> {len(kept)} kept ({kept.dedup_method})")
        print(f"                {files['manifest']}")
        report.results.context["candidate_manifest"] = str(files["manifest"])
    except Exception as exc:  # export is a by-product; L0 itself must still be written
        print(f"candidates    : export failed ({type(exc).__name__}: {exc})")
        report.results.context["candidate_manifest"] = None
    out = report.results.save(paths.result_path("L0_diagnostic.json"))
    print(f"wrote {out}")
    print(report.results.to_frame().to_string(index=False, max_colwidth=44))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="both", choices=["both", "audit", "diagnostic"])
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N frames of each set")
    ap.add_argument("--candidates", type=int, default=20, help="worst diagnostic frames considered for export")
    ap.add_argument("--refresh", action="store_true", help="ignore the prediction cache")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--allow-login-node", action="store_true",
                    help="bypass the compute-node check; only for trivial smoke tests")
    args = ap.parse_args()
    require_compute_node("L0: model evaluation on the phase-1 label sets", allow_login_node=args.allow_login_node)

    model = build_model(threads=args.threads, device=args.device)
    print(f"model         : {model.spec.name}  sha256={model.spec.checkpoint_sha256[:16]}  dtype={model.spec.dtype}")
    if args.set in ("both", "audit"):
        run_audit(model, args)
    if args.set in ("both", "diagnostic"):
        run_diagnostic(model, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
