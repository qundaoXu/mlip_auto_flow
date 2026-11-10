"""Canonical filesystem locations for the phase-1 GST225 benchmark.

Everything outside ``eval/`` is treated as read-only reference material.  The
only writable roots are :data:`RESULTS` and :data:`DATA`.
"""

from __future__ import annotations

import os
from pathlib import Path

# ``eval/scripts/mlipeval/paths.py`` -> ``eval/scripts`` -> ``eval`` -> workspace
EVAL = Path(__file__).resolve().parents[2]
WORKSPACE = EVAL.parent

# --- writable -------------------------------------------------------------
SCRIPTS = EVAL / "scripts"
NOTEBOOKS = EVAL / "notebooks"

#: Model namespace.  Five models run the identical drivers, so their outputs are
#: separated by ``MLIPEVAL_MODEL``; with the variable unset the layout is the
#: flat phase-1 one that the notebooks read.
MODEL_KEY = os.environ.get("MLIPEVAL_MODEL", "").strip()

RESULTS = (EVAL / "results" / "models" / MODEL_KEY) if MODEL_KEY else (EVAL / "results")
DATA = (EVAL / "data" / "models" / MODEL_KEY) if MODEL_KEY else (EVAL / "data")

#: Prediction / memo cache.  Deliberately *not* namespaced: every key already
#: contains the checkpoint sha256 and the input fingerprint, so five models
#: share it without collision, and reference-only memos are computed once.
CACHE = EVAL / "data" / "cache"

#: Reference data harvested from the read-only VASP tree into a form the
#: framework reads quickly.  Not namespaced either: it is model-independent, and
#: a per-model copy would let two models be scored against different harvests.
REFERENCE = EVAL / "data" / "reference"
H5_LABELS = REFERENCE / "phase1_h5_relabel.extxyz"
H5_LEAKAGE = REFERENCE / "phase1_h5_relabel.leakage.json"
DIAGNOSTIC_LABELS = REFERENCE / "phase1_static_diagnostic.extxyz"
DIAGNOSTIC_LEAKAGE = REFERENCE / "phase1_static_diagnostic.leakage.json"

#: Roots that hold one subdirectory per model, for the cross-model report.
MODEL_RESULTS_ROOT = EVAL / "results" / "models"
MODEL_DATA_ROOT = EVAL / "data" / "models"

# --- read-only reference --------------------------------------------------
GST_REF = WORKSPACE / "GST225_ref"
AIMD_216 = GST_REF / "AIMD/216atoms"
AIMD_306 = GST_REF / "AIMD/306atoms"
LABELLED_615 = GST_REF / "single_point/GST225_init_noStress.xyz"
CUBIC_SEEDS = GST_REF / "cubic_phase"
MP_STRUCTURES = GST_REF / "mp_structures"

PHASE1 = WORKSPACE / "phase1_vasp_inputs_20260828"
TASK_MANIFEST = PHASE1 / "task_manifest.csv"

MODEL_ROOT = WORKSPACE / "MLIP_run/universal_MLIP"
MACE_MPA0 = MODEL_ROOT / "mace-mpa-0-medium.model"

EXTERNAL_ACE = WORKSPACE / "database/ACE_potential_and_training_dataset/GST-ACE-24_training_dataset.xyz"
EXTERNAL_NE = WORKSPACE / "database/processed_NE_dataset.xyz"


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
    if resolved.is_relative_to(EVAL):
        raise RuntimeError(f"{resolved} is inside eval/, not a read-only reference")
    return resolved


def n_threads(default: int = 32) -> int:
    """Thread count for torch / numpy heavy sections."""
    env = os.environ.get("MLIPEVAL_THREADS")
    if env:
        return int(env)
    return min(default, len(os.sched_getaffinity(0)))
