"""Where the two halves of the software live.

The CLI is a thin layer: it owns no analysis code.  It finds

* the evaluation framework (``eval_framework/``, the ``mlipeval`` package plus
  the ``run_*.py`` drivers) through ``MLIPEVAL_HOME``, and
* the active-learning core script (``activate_learning_full_flow.py`` plus
  ``submitjobs.sh`` / ``outcar2xyz.sh``) through ``ALFLOW_HOME``,

and falls back to the in-tree layout when the variables are unset.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

RELEASE_ROOT = Path(__file__).resolve().parents[1]


def _first_existing(candidates: list[Path]) -> Path | None:
    for c in candidates:
        if c.exists():
            return c.resolve()
    return None


def eval_home() -> Path:
    env = os.environ.get("MLIPEVAL_HOME")
    found = _first_existing(
        ([Path(env)] if env else [])
        + [RELEASE_ROOT / "eval_framework", RELEASE_ROOT.parent / "eval_framework"]
    )
    if found is None or not (found / "scripts" / "mlipeval").is_dir():
        sys.exit(
            "mlipeval: cannot find the evaluation framework. Set MLIPEVAL_HOME to the "
            "directory that contains scripts/mlipeval/ (e.g. .../eval_framework)."
        )
    return found


def eval_scripts() -> Path:
    return eval_home() / "scripts"


def alflow_home() -> Path:
    env = os.environ.get("ALFLOW_HOME")
    found = _first_existing(
        ([Path(env)] if env else [])
        + [RELEASE_ROOT / "al_flow", RELEASE_ROOT.parent / "al_flow"]
    )
    if found is None or not (found / "activate_learning_full_flow.py").exists():
        sys.exit(
            "alflow: cannot find activate_learning_full_flow.py. Set ALFLOW_HOME to the "
            "directory that contains it (together with submitjobs.sh and outcar2xyz.sh)."
        )
    return found


def import_mlipeval():
    """Put ``eval_framework/scripts`` on ``sys.path`` and return the package."""
    scripts = str(eval_scripts())
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import mlipeval  # noqa: WPS433  (import after path setup is the point)

    return mlipeval
