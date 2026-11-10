"""Canonical filesystem locations for the phase-1 GST225 benchmark.

Everything outside ``eval/`` is treated as read-only reference material.  The
only writable roots are :data:`RESULTS` and :data:`DATA`.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import config

# ``eval/scripts/mlipeval/paths.py`` -> ``eval/scripts`` -> ``eval``
EVAL = Path(__file__).resolve().parents[2]
#: Root of the read-only reference tree and model runs (``paths.workspace`` in
#: the site config / ``$MLIPEVAL_WORKSPACE``; default: the parent of ``eval/``).
WORKSPACE = config.workspace()
#: Where ``results/`` and ``data/`` are written (``paths.output``; default ``eval/``).
OUTPUT = config.output_root()

# --- writable -------------------------------------------------------------
SCRIPTS = EVAL / "scripts"
NOTEBOOKS = EVAL / "notebooks"

#: Model namespace.  Five models run the identical drivers, so their outputs are
#: separated by ``MLIPEVAL_MODEL``; with the variable unset the layout is the
#: flat phase-1 one that the notebooks read.
MODEL_KEY = os.environ.get("MLIPEVAL_MODEL", "").strip()

RESULTS = (OUTPUT / "results" / "models" / MODEL_KEY) if MODEL_KEY else (OUTPUT / "results")
DATA = (OUTPUT / "data" / "models" / MODEL_KEY) if MODEL_KEY else (OUTPUT / "data")

#: Prediction / memo cache.  Deliberately *not* namespaced: every key already
#: contains the checkpoint sha256 and the input fingerprint, so five models
#: share it without collision, and reference-only memos are computed once.
CACHE = OUTPUT / "data" / "cache"

#: Reference data harvested from the read-only VASP tree into a form the
#: framework reads quickly.  Not namespaced either: it is model-independent, and
#: a per-model copy would let two models be scored against different harvests.
REFERENCE = OUTPUT / "data" / "reference"
H5_LABELS = REFERENCE / "phase1_h5_relabel.extxyz"
H5_LEAKAGE = REFERENCE / "phase1_h5_relabel.leakage.json"
DIAGNOSTIC_LABELS = REFERENCE / "phase1_static_diagnostic.extxyz"
DIAGNOSTIC_LEAKAGE = REFERENCE / "phase1_static_diagnostic.leakage.json"

#: Roots that hold one subdirectory per model, for the cross-model report.
MODEL_RESULTS_ROOT = OUTPUT / "results" / "models"
MODEL_DATA_ROOT = OUTPUT / "data" / "models"

# --- read-only reference --------------------------------------------------
GST_REF = config.path("paths.reference.gst_ref")
AIMD_216 = GST_REF / "AIMD/216atoms"
AIMD_306 = GST_REF / "AIMD/306atoms"
LABELLED_615 = GST_REF / "single_point/GST225_init_noStress.xyz"
CUBIC_SEEDS = GST_REF / "cubic_phase"
MP_STRUCTURES = GST_REF / "mp_structures"

PHASE1 = config.path("paths.reference.phase1")
TASK_MANIFEST = PHASE1 / "task_manifest.csv"

MODEL_ROOT = config.path("paths.models.universal")
SPECIALISED_ROOT = config.path("paths.models.specialised")
AL_LADDER_ROOT = config.path("paths.models.al_ladder")
GST_AGING_ROOT = config.path("paths.models.gst_aging")
MACE_MPA0 = MODEL_ROOT / "mace-mpa-0-medium.model"

EXTERNAL_ACE = config.path("paths.training_sets.gst_ace24")
EXTERNAL_NE = config.path("paths.training_sets.processed_ne")
GST_AGING_DATASET = config.path("paths.training_sets.gst_aging")
AL_LADDER_TRAIN_SETS = config.path("paths.training_sets.al_ladder")

# --- external tools ----------------------------------------------------------
LAMMPS_EXE = config.path("lammps.executable")
LAMMPS_COMMON = config.path("lammps.common_dir")
AL_LADDER_EXPORT = config.path("lammps.al_ladder_export")


def ensure_writable_dirs() -> None:
    for path in (RESULTS, DATA):
        path.mkdir(parents=True, exist_ok=True)


def result_path(*parts: str) -> Path:
    """Path under ``eval/results`` with parent directories created."""
    path = RESULTS.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def md_result_path(name: str) -> Path:
    """依赖 MD 的结果文件路径，带上引擎标记：``L4_x.json`` -> ``L4_x__lammps.json``。

    L0/L1/L2 不经过 MD，用 :func:`result_path` 即可——给它们也加引擎后缀
    只会凭空造出两份一模一样的文件。只有 L3/L4/L5 的数字随引擎变。
    """
    from .state_points import current_engine  # 延迟导入：state_points 依赖本模块

    stem, dot, ext = name.rpartition(".")
    return result_path(f"{stem}__{current_engine()}{dot}{ext}")


def data_path(*parts: str) -> Path:
    """Path under ``eval/data`` with parent directories created.

    ``cache`` is routed to the shared, un-namespaced cache root (see
    :data:`CACHE`); everything else follows the per-model namespace.
    """
    root = CACHE if parts and parts[0] == "cache" else DATA
    path = root.joinpath(*(parts[1:] if root is CACHE else parts))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def assert_read_only(path: Path) -> Path:
    """Guard against accidentally writing outside ``eval/``."""
    resolved = Path(path).resolve()
    if resolved.is_relative_to(EVAL) or resolved.is_relative_to(OUTPUT):
        raise RuntimeError(f"{resolved} is inside eval/, not a read-only reference")
    return resolved


def n_threads(default: int = 32) -> int:
    """Thread count for torch / numpy heavy sections."""
    env = os.environ.get("MLIPEVAL_THREADS")
    if env:
        return int(env)
    return min(default, len(os.sched_getaffinity(0)))
