"""On-disk memoisation for trajectory-derived arrays.

The L3 primitives (partial RDFs, angular distributions, coordination
histograms, rings, ALTBC, S(q)) are pure functions of a trajectory segment and a
frozen :class:`~mlipeval.neighbors.StructureParams`.  They are also the slowest
part of the notebooks: recomputing them turns a re-read of nb04 into several
minutes of single-core work, which is exactly what should not happen on a shared
login node.

Memoising them keeps the notebooks instructive -- they still call the analysis
functions, so a reader sees the real API -- while making a re-run a file read.
The key covers everything that can change the answer: the function, the
trajectory contents, and every post-processing knob.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .paths import data_path

#: Bump when a primitive's definition changes, to invalidate old entries.
MEMO_VERSION = "1"

_ENABLED = True


def set_enabled(enabled: bool) -> None:
    """Turn memoisation off (for timing the underlying computation)."""
    global _ENABLED
    _ENABLED = enabled


def trajectory_identity(traj) -> dict[str, Any]:
    """Cheap but faithful identity of a trajectory segment.

    A strided sample of the positions is hashed rather than the whole array:
    two different segments of the same run must not collide, but hashing 11 MB
    on every call would defeat the purpose.
    """
    positions = np.asarray(traj.positions)
    step = max(1, positions.shape[0] // 24)
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(positions[::step], dtype=np.float32).tobytes())
    digest.update(np.ascontiguousarray(traj.cells[0], dtype=np.float32).tobytes())
    digest.update("".join(traj.symbols).encode())
    return {
        "name": traj.name,
        "n_frames": int(len(traj)),
        "n_atoms": int(traj.n_atoms),
        "sample_stride": int(traj.sample_stride),
        "positions_sha": digest.hexdigest()[:16],
    }


def _key(name: str, identity: Mapping[str, Any]) -> str:
    blob = json.dumps({"v": MEMO_VERSION, "fn": name, **identity}, sort_keys=True, default=str)
    return f"{name}_{hashlib.sha256(blob.encode()).hexdigest()[:20]}"


def cached_arrays(
    name: str,
    identity: Mapping[str, Any],
    compute: Callable[[], Any],
    *,
    refresh: bool = False,
) -> Any:
    """Memoise a dict of arrays, or a bare array, on disk.

    Values that are not arrays (scalars, strings, small nested dicts) are stored
    in a JSON sidecar inside the same ``.npz``, so a primitive that mixes them --
    ``structure_factor_direct`` returns arrays plus a method string -- round-trips
    unchanged.
    """
    if not _ENABLED:
        return compute()

    path = data_path("cache", "analysis", f"{_key(name, identity)}.npz")
    if path.exists() and not refresh:
        try:
            return _load(path)
        except Exception:
            pass  # a corrupt or stale entry is simply recomputed

    value = compute()
    try:
        _save(path, value)
    except Exception:
        pass
    return value


def _save(path: Path, value: Any) -> None:
    if isinstance(value, np.ndarray):
        np.savez_compressed(path, __bare__=value, __meta__=json.dumps({"kind": "array"}))
        return
    if not isinstance(value, Mapping):
        raise TypeError(f"cannot memoise {type(value)!r}")
    arrays: dict[str, np.ndarray] = {}
    extras: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, np.ndarray):
            arrays[f"a__{key}"] = item
        elif isinstance(item, (int, float, str, bool, type(None))):
            extras[key] = item
        else:
            extras[key] = json.loads(json.dumps(item, default=_jsonable))
    np.savez_compressed(path, __meta__=json.dumps({"kind": "mapping", "extras": extras}), **arrays)


def _load(path: Path) -> Any:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["__meta__"]))
        if meta["kind"] == "array":
            return z["__bare__"]
        out: dict[str, Any] = dict(meta["extras"])
        for key in z.files:
            if key.startswith("a__"):
                out[key[3:]] = z[key]
        return out


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return str(obj)
