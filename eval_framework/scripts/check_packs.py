#!/usr/bin/env python
"""Round-trip check of evaluation packs: build, rescore, compare with the HPC files.

    python check_packs.py --reference gst225_phase1_v1.mlipref --out packs/ [--models a,b] [--json report.json]

For every model with results under ``results/models``: build its pack from the
files already on disk, rescore L0/L1/L2 on this machine, and compare every
metric with the result file the HPC driver wrote.  Exit status 1 if any model
disagrees beyond the fit tolerance.  Pure file reading; safe on a login node.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mlipeval import pack, paths, rescore


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", default=None, help="reference pack (default: $MLIPEVAL_REFERENCE)")
    ap.add_argument("--out", required=True, help="directory for the model packs")
    ap.add_argument("--models", default=None, help="comma-separated keys (default: every model with results)")
    ap.add_argument("--json", default=None, help="write the full report here")
    args = ap.parse_args()

    ref = pack.open_reference(args.reference or pack.default_reference_path())
    keys = (args.models.split(",") if args.models
            else sorted(p.name for p in paths.MODEL_RESULTS_ROOT.iterdir() if p.is_dir()))
    out_dir = Path(args.out)
    report, failed = {}, []
    print(f"reference {ref.reference_id} (complete={ref.manifest.get('complete')})")
    print(f"{'model':<18} {'L0 audit':>10} {'L0 diag':>10} {'L1':>8} {'L2':>8}  gate  notes")
    for key in keys:
        try:
            path, summary = pack.build_model_pack(key, out_dir / f"{key}.mlippack", reference=ref)
            out = rescore.rescore(pack.open_model_pack(path), ref)
            cmp = rescore.compare_with_pack(out)
        except Exception as exc:  # report and move on: one bad model must not hide the rest
            print(f"{key:<18} ERROR {type(exc).__name__}: {exc}")
            failed.append(key)
            report[key] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        cells = []
        for stem in ("L0_audit", "L0_diagnostic", "L1", "L2"):
            c = cmp.get(stem)
            if c is None:
                cells.append("—")
            elif c.get("identical") is None:
                cells.append("no orig")
            else:
                cells.append(f"{c['n_agree']}/{c['n_compared']}" + ("" if c["identical"] else "*"))
        gate = out.results["L1"].context["gate"]["pass"] if "L1" in out.results else None
        notes = [w["id"] for w in out.verification["checks"] if not w["ok"]]
        notes += [f"absent:{a}" for a in summary["absent"]]
        print(f"{key:<18} {cells[0]:>10} {cells[1]:>10} {cells[2]:>8} {cells[3]:>8}  "
              f"{'pass' if gate else 'shut' if gate is False else '—':<4}  {', '.join(notes)}")
        bad = [s for s, c in cmp.items() if c.get("identical") is False]
        if bad:
            failed.append(key)
        report[key] = {"pack": str(path), "summary": summary, "verification": out.verification,
                       "comparison": cmp}
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"\n{len(keys) - len(failed)}/{len(keys)} models reproduce their HPC result files"
          + (f"; differing: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
