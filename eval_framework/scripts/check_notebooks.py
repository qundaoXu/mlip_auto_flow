#!/usr/bin/env python
"""Execute every notebook's code cells and report which ones run clean.

    python check_notebooks.py [--only nb04] [--list]

The ``mace`` environment has no jupyter, so this runs each notebook's code cells
as a plain script in the notebook's own directory, with a ``display`` shim and a
headless matplotlib backend.  It is the reproducible check that the delivered
notebooks actually execute against the current results, rather than a claim that
they do.

Like the other MACE-touching drivers this belongs in a PBS job, not on the login
node: several notebooks build the model and evaluate potential energy curves.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mlipeval.paths import NOTEBOOKS
from mlipeval.pbs import describe_environment, require_compute_node


#: Any CJK glyph inside a figure is a bug: the deliverable convention is English
#: figure text with Chinese prose, and DejaVu Sans cannot render CJK anyway --
#: matplotlib emits "Glyph ... missing from font(s)" and draws a tofu box.
_CJK = re.compile(r"[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uff00-\uffef]")


def cjk_texts_in_open_figures() -> list[str]:
    """Every string drawn into a currently open figure that contains CJK.

    Walking the Text artists catches text no grep can find: titles passed into a
    plotting helper, tick labels taken from a DataFrame column, legend entries
    built from dict keys.
    """
    import matplotlib.text

    found: list[str] = []
    for number in plt.get_fignums():
        figure = plt.figure(number)
        for artist in figure.findobj(matplotlib.text.Text):
            content = artist.get_text()
            if content and _CJK.search(content):
                found.append(content)
    return found


def _display(*objects) -> None:
    """Stand-in for IPython's ``display`` so notebook cells run unchanged."""
    import pandas as pd

    for obj in objects:
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            print(obj.to_string()[:2000])
        else:
            print(repr(obj)[:1000])


def run_notebook(path: Path) -> dict[str, object]:
    cells = [c for c in json.loads(path.read_text())["cells"] if c["cell_type"] == "code"]
    namespace: dict[str, object] = {"__name__": "__main__", "display": _display}
    figures = 0
    cjk: list[str] = []
    started = time.time()
    for index, cell in enumerate(cells, 1):
        source = "".join(cell["source"])
        try:
            exec(compile(source, f"{path.name}:cell{index}", "exec"), namespace)
        except Exception:
            return {
                "notebook": path.name,
                "ok": False,
                "failed_cell": index,
                "n_cells": len(cells),
                "figures": figures,
                "seconds": round(time.time() - started, 1),
                "traceback": traceback.format_exc(limit=6),
                "source_head": source.split("\n")[0][:120],
            }
        figures += len(plt.get_fignums())
        for text in cjk_texts_in_open_figures():
            cjk.append(f"cell{index}: {text}")
        for number in plt.get_fignums():
            plt.close(number)
    return {
        "notebook": path.name,
        "ok": True,
        "n_cells": len(cells),
        "figures": figures,
        "cjk_in_figures": cjk,
        "seconds": round(time.time() - started, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None, help="substring of the notebook filename")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--allow-login-node", action="store_true")
    args = ap.parse_args()

    notebooks = sorted(NOTEBOOKS.glob("nb*.ipynb"))
    if args.only:
        notebooks = [n for n in notebooks if args.only in n.name]
    if args.list:
        for n in notebooks:
            print(n.name)
        return 0

    require_compute_node("notebook execution check", allow_login_node=args.allow_login_node)
    print(json.dumps(describe_environment(), indent=2))

    original = Path.cwd()
    reports = []
    for notebook in notebooks:
        import os

        os.chdir(NOTEBOOKS)
        print(f"\n{'=' * 78}\n{notebook.name}\n{'=' * 78}", flush=True)
        report = run_notebook(notebook)
        reports.append(report)
        os.chdir(original)
        status = "OK" if report["ok"] else f"FAILED at cell {report['failed_cell']}"
        print(f"\n-> {status}   {report['n_cells']} cells, {report['figures']} figures, "
              f"{report['seconds']}s", flush=True)
        if not report["ok"]:
            print(report["traceback"])
        for text in report.get("cjk_in_figures", []):
            print(f"   CJK IN FIGURE  {text}")

    print(f"\n{'=' * 78}")
    n_cjk_total = 0
    for r in reports:
        mark = "ok " if r["ok"] else "FAIL"
        n_cjk = len(r.get("cjk_in_figures", []))
        n_cjk_total += n_cjk
        flag = f"  CJK×{n_cjk}" if n_cjk else ""
        print(f"  [{mark}] {r['notebook']:46s} {r['n_cells']:>2} cells  "
              f"{r['figures']:>2} figures  {r['seconds']:>6.1f}s{flag}")
    n_bad = sum(1 for r in reports if not r["ok"])
    print(f"\n{len(reports) - n_bad}/{len(reports)} notebooks execute clean")
    print(f"{n_cjk_total} CJK strings found inside figures "
          f"(must be 0: figure text is English, prose is Chinese)")
    return 1 if (n_bad or n_cjk_total) else 0


if __name__ == "__main__":
    raise SystemExit(main())
