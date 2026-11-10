"""Site configuration: where things live and how jobs are launched.

Nothing site-specific is hard-coded in the framework any more.  Every path,
conda location, LAMMPS binary and scheduler setting is read from one file,
``mlip_auto_flow.yaml`` (YAML or JSON), with environment variables taking
precedence over it and built-in defaults filling the rest::

    environment variable  >  config file  >  default

The file is found by, in order: ``$MLIPEVAL_CONFIG``; ``./mlip_auto_flow.yaml``
in the current directory; ``mlip_auto_flow.yaml`` at the repository root.  None
of them is required -- with no file at all the defaults reproduce the original
workspace layout (reference data and model runs beside ``eval_framework/``).

A template listing every key is ``config/mlip_auto_flow.example.yaml``; the
``mlipeval config`` command prints the resolved values and checks that each
path exists.

This module imports nothing from the rest of :mod:`mlipeval`, so
:mod:`mlipeval.paths` can depend on it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

#: ``eval_framework/scripts/mlipeval/config.py`` -> ``eval_framework``
EVAL_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = EVAL_DIR.parent
CONFIG_NAME = "mlip_auto_flow.yaml"

#: Short environment-variable names for the settings people change most.  Any
#: other scalar can still be overridden as ``MLIPEVAL_<SECTION>_<KEY>``
#: (dots -> underscores, upper case), e.g. ``MLIPEVAL_SCHEDULER_WALLTIME``.
ENV_ALIASES = {
    "paths.workspace": "MLIPEVAL_WORKSPACE",
    "paths.output": "MLIPEVAL_OUTPUT",
    "environment.conda_root": "MLIPEVAL_CONDA_ROOT",
    "environment.conda_envs": "MLIPEVAL_CONDA_ENVS",
    "lammps.executable": "MLIPEVAL_LMP",
    "scheduler.project": "MLIPEVAL_PBS_PROJECT",
    "scheduler.queue": "MLIPEVAL_PBS_QUEUE",
}

#: Defaults.  Relative paths are resolved against ``paths.workspace`` (or, for
#: ``paths.workspace`` itself, against the parent of ``eval_framework``).
DEFAULTS: dict[str, Any] = {
    "paths": {
        "workspace": None,  # default: parent directory of eval_framework/
        "output": None,  # default: eval_framework/ (results/, data/ go here)
        "reference": {
            "gst_ref": "GST225_ref",
            "phase1": "phase1_vasp_inputs_20260828",
        },
        "models": {
            "universal": "MLIP_run/universal_MLIP",
            "specialised": "MLIP_run/specialized",
            "al_ladder": "MLIP_run/specialized/al_ladder",
            # outside the workspace on the original cluster: <workspace>/../myproject/1-GST225-aging
            "gst_aging": "../myproject/1-GST225-aging",
        },
        "training_sets": {
            "gst_ace24": "database/ACE_potential_and_training_dataset/GST-ACE-24_training_dataset.xyz",
            "processed_ne": "database/processed_NE_dataset.xyz",
            "gst_aging": "../myproject/1-GST225-aging/dataset",
            "al_ladder": "MLIP_run/specialized/al_ladder/train_sets",
        },
    },
    "environment": {
        "conda_root": None,  # default: from $CONDA_EXE, else <workspace>/../miniconda3
        "conda_envs": None,  # default: <conda_root>/envs
        # backend -> conda environment (a name under conda_envs, or an absolute path)
        "backend_envs": {
            "mace": "mace",
            "nequip": "nequip",
            "deepmd": "deepmd",
            "mattersim": "gnnp",
            "chgnet": "gnnp",
        },
    },
    "lammps": {
        "executable": "MLIP_run/universal_MLIP/_build/advancesoft_build/lmp",
        "common_dir": "MLIP_run/universal_MLIP/common",
        "al_ladder_export": "MLIP_run/specialized/al_ladder/export_mliap.sh",
    },
    "scheduler": {
        "kind": "pbs",
        "project": None,  # PBS -P project code; omitted from job scripts when unset
        "queue": None,
        "select": "1:ncpus=12:mpiprocs=1:ompthreads=12:mem=225gb:ngpus=1",
        "walltime": "12:00:00",  # write_job default; submit_pbs.py sets per-stage walltimes
        "modules": ["gcc/13.1.0", "openmpi/gcc/64/4.1.8"],
        "require_compute_node": True,
    },
    # Extra or overriding model-registry entries, keyed by model key; see the
    # example config for the fields.  Built-in entries stay available.
    "models": {},
}


# --------------------------------------------------------------------- loading
def config_file() -> Path | None:
    env = os.environ.get("MLIPEVAL_CONFIG")
    if env:
        p = Path(os.path.expandvars(env)).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"MLIPEVAL_CONFIG points to a missing file: {p}")
        return p
    for cand in (Path.cwd() / CONFIG_NAME, REPO_ROOT / CONFIG_NAME):
        if cand.exists():
            return cand
    return None


def _read(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text) or {}
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - pyyaml is a declared dependency
        raise ImportError(f"reading {path} needs pyyaml (pip install pyyaml), or use a .json file") from exc
    return yaml.safe_load(text) or {}


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _load() -> tuple[dict, Path | None]:
    path = config_file()
    data = _read(path) if path else {}
    # a combined file may hold the AL settings too; they are not ours
    data = {k: v for k, v in data.items() if k != "alflow"}
    return _merge(DEFAULTS, data), path


SETTINGS, SOURCE = _load()


# --------------------------------------------------------------------- access
def _env_name(key: str) -> str:
    return ENV_ALIASES.get(key) or "MLIPEVAL_" + key.replace(".", "_").upper()


def get(key: str, default: Any = None) -> Any:
    """Dotted lookup, e.g. ``get("scheduler.project")``; env var wins."""
    env = os.environ.get(_env_name(key))
    if env is not None and env != "":
        return env
    node: Any = SETTINGS
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return default if node is None else node


def _expand(value: str | os.PathLike) -> Path:
    return Path(os.path.expandvars(str(value))).expanduser()


def workspace() -> Path:
    raw = get("paths.workspace")
    if raw is None:
        return EVAL_DIR.parent.resolve()
    p = _expand(raw)
    return (p if p.is_absolute() else (EVAL_DIR.parent / p)).resolve()


def path(key: str, base: Path | None = None) -> Path | None:
    """A configured path; relative values resolve against the workspace."""
    raw = get(key)
    if raw is None:
        return None
    p = _expand(raw)
    if p.is_absolute():
        return p
    return ((base or workspace()) / p).resolve()


def output_root() -> Path:
    return path("paths.output") or EVAL_DIR


def conda_root() -> Path:
    raw = get("environment.conda_root")
    if raw is not None:
        return _expand(raw)
    exe = os.environ.get("CONDA_EXE")
    if exe:
        return Path(exe).resolve().parents[1]
    return (workspace().parent / "miniconda3").resolve()


def conda_envs() -> Path:
    raw = get("environment.conda_envs")
    return _expand(raw) if raw is not None else conda_root() / "envs"


def conda_env(name_or_path: str | os.PathLike) -> Path:
    """A conda env given by name (under :func:`conda_envs`) or absolute path."""
    p = _expand(name_or_path)
    return p if p.is_absolute() else conda_envs() / p


def backend_env(backend: str) -> Path:
    envs = get("environment.backend_envs", {}) or {}
    return conda_env(envs.get(backend, backend))


def summary() -> dict[str, Any]:
    """Resolved settings, for ``mlipeval config``."""
    keys = [
        "paths.reference.gst_ref", "paths.reference.phase1",
        "paths.models.universal", "paths.models.specialised", "paths.models.al_ladder",
        "paths.models.gst_aging",
        "paths.training_sets.gst_ace24", "paths.training_sets.processed_ne",
        "paths.training_sets.gst_aging", "paths.training_sets.al_ladder",
        "lammps.executable", "lammps.common_dir", "lammps.al_ladder_export",
    ]
    out: dict[str, Any] = {
        "config_file": str(SOURCE) if SOURCE else None,
        "paths.workspace": str(workspace()),
        "paths.output": str(output_root()),
        "environment.conda_root": str(conda_root()),
        "environment.conda_envs": str(conda_envs()),
    }
    for k in keys:
        out[k] = str(path(k))
    for backend in (get("environment.backend_envs", {}) or {}):
        out[f"environment.backend_envs.{backend}"] = str(backend_env(backend))
    for k in ("kind", "project", "queue", "select", "walltime", "modules", "require_compute_node"):
        out[f"scheduler.{k}"] = get(f"scheduler.{k}")
    out["models (from config)"] = sorted((get("models", {}) or {}).keys())
    return out
