"""Loaders and a shared palette for the cross-model comparison notebook.

Every per-model driver writes into ``data/models/<key>/`` and
``results/models/<key>/`` (see :mod:`mlipeval.paths`).  A notebook that compares
five models therefore cannot use ``paths.DATA``, which is fixed at import time to
whichever model ``MLIPEVAL_MODEL`` names -- it needs to reach into all five
trees.  These helpers do exactly that and nothing else: no model is loaded, no
GPU is touched, so nb07 stays a login-node notebook like the other six.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from . import models as model_registry
from .paths import CACHE, MODEL_DATA_ROOT, MODEL_RESULTS_ROOT

#: One colour per model, used by every panel so a reader can follow one model
#: across the whole notebook without re-reading legends.
MODEL_COLORS = {
    "mace_mpa0": "#c1121f",
    "nequip_oam_l": "#2e86ab",
    "dpa4_pro": "#e0a800",
    "mattersim_5m": "#3f8f5b",
    "chgnet_030": "#7b4b94",
}
REFERENCE_COLOR = "#222222"

#: Ramp for the 0-100 relative scores.  Deliberately a *single* hue, light to
#: dark: the score is a magnitude with no meaningful midpoint (50 does not mean
#: "neutral", it means "about twice the best model's error"), so a red-yellow-green
#: map would put a hue break at 50 and invent a boundary that is not in the data.
#: Monotone lightness also survives greyscale printing and colour-vision
#: deficiency without any per-step checking.  Indigo is free of all five model
#: identity colours above, so a cell can never be mistaken for a series colour.
#: Stepped so lightness falls *evenly* and stops short of near-black.  The first
#: attempt ended at #1c1f45 and every score from 85 to 100 rendered as the same
#: navy -- which is exactly the band this comparison lives in, because a score of
#: 100 is awarded per level and most models land within 15 points of it.
_SCORE_STEPS = ["#f7f8fc", "#dee2f2", "#c2c9e6", "#a2acd6", "#8290c2", "#6373aa",
                "#4a5a90"]
SCORE_CMAP = LinearSegmentedColormap.from_list("mlipeval_score", _SCORE_STEPS)


#: The two inks a cell may use.  Text never wears the ramp colour itself.
_DARK_INK, _LIGHT_INK = "#111111", "#ffffff"


def _relative_luminance(rgb) -> float:
    """WCAG relative luminance: gamma-expanded, not the raw sRGB weighted sum."""
    chan = [(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4) for c in rgb[:3]]
    return 0.2126 * chan[0] + 0.7152 * chan[1] + 0.0722 * chan[2]


def _contrast(a, b) -> float:
    la, lb = sorted((_relative_luminance(a), _relative_luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def score_ink(value: float, vmin: float = 0.0, vmax: float = 100.0) -> str:
    """Readable text colour for a cell filled with :data:`SCORE_CMAP`.

    Picks whichever ink actually contrasts more against that cell, computed on
    WCAG relative luminance.  An earlier version thresholded the *raw* sRGB
    weighted sum at 0.55, which flips to white too early: four cells in the
    L0-L5 matrix (scores 71-78) came out at 3.5:1, under the 4.5:1 AA floor.
    """
    bg = SCORE_CMAP((float(value) - vmin) / (vmax - vmin))[:3]
    dark = _contrast((0.067, 0.067, 0.067), bg)
    light = _contrast((1.0, 1.0, 1.0), bg)
    return _DARK_INK if dark >= light else _LIGHT_INK


#: Marker per model, so the figures survive being printed in greyscale.
MODEL_MARKERS = {
    "mace_mpa0": "o",
    "nequip_oam_l": "s",
    "dpa4_pro": "^",
    "mattersim_5m": "D",
    "chgnet_030": "v",
}


def label(model_key: str) -> str:
    return model_registry.get(model_key).label


def color(model_key: str) -> str:
    return MODEL_COLORS.get(model_key, "#777777")


def marker(model_key: str) -> str:
    return MODEL_MARKERS.get(model_key, "o")


def available_models() -> list[str]:
    """Registered models that actually have results on disk, in display order."""
    return [k for k in model_registry.ORDER if (MODEL_RESULTS_ROOT / k).is_dir()]


def data_file(model_key: str, name: str) -> Path:
    return MODEL_DATA_ROOT / model_key / name


def load_npz(model_key: str, name: str) -> dict[str, Any] | None:
    """One model's saved arrays, or ``None`` when that stage did not run."""
    path = data_file(model_key, name)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def pec_curves(model_key: str) -> dict[str, dict[str, np.ndarray]] | None:
    """The six binary potential energy curves, from the shared PEC cache.

    The cache filename carries the checkpoint hash, the artefact name, the
    species set and the grid; only the hash is stable enough to match on here,
    because entries written before the artefact joined the key are still valid
    (the same weights produced them).
    """
    spec_sha = _checkpoint_sha(model_key)
    if spec_sha is None:
        return None
    hits = sorted(CACHE.glob(f"pec2__{spec_sha[:10]}__*.npz"))
    if not hits:
        return None
    with np.load(hits[-1], allow_pickle=False) as z:
        pairs = [str(p) for p in z["pairs"]]
        out = {}
        for i, pair in enumerate(pairs):
            out[pair] = {
                "r": z["r"],
                "energy": z["energy"][i],
                "force": z["force_analytic"][i],
                "monotonic_repulsive": bool(z["monotonic_repulsive"][i]),
                "max_force_mismatch": float(z["max_force_mismatch"][i]),
                "convergence_ratio": float(z["convergence_ratio"][i]),
            }
    return out


def _checkpoint_sha(model_key: str) -> str | None:
    """Checkpoint sha256 as recorded in that model's own result files.

    Reading it back from a result file rather than re-hashing the checkpoint
    keeps this notebook free of any filesystem work on 80-300 MB model files.
    """
    import json

    for path in sorted((MODEL_RESULTS_ROOT / model_key).glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        sha = (payload.get("model") or {}).get("checkpoint_sha256")
        if sha:
            return str(sha)
    return None


def metric_table(model_keys, pattern: str, files: tuple[str, ...], field: str = "value"):
    """Long-form table of one metric family across models, for quick plotting."""
    import re

    import pandas as pd

    from .ranking import _field_value, _matching_stems, load_results

    rx = re.compile(pattern)
    rows = []
    for key in model_keys:
        results = load_results(key)
        stems: list[str] = []
        for spec in files:
            stems.extend(s for s in _matching_stems(spec, results) if s not in stems)
        for stem in stems:
            for record in results[stem].get("results", []):
                if not rx.search(record["metric_id"]):
                    continue
                rows.append({
                    "model": key,
                    "label": label(key),
                    "metric_id": record["metric_id"],
                    "value": _field_value(record, field),
                    "unit": record.get("unit"),
                    "status": record.get("status"),
                })
    return pd.DataFrame(rows)

def metric_record(model_key: str, metric_id: str,
                  files: tuple[str, ...] = ()) -> dict[str, Any] | None:
    """One model's full record for one metric id, ``extra`` included.

    :func:`metric_table` flattens to a value column, which is right for plotting
    a family of metrics but loses ``extra`` -- and some metrics carry their real
    payload there (the phase ladder stores both energy ladders under
    ``extra``, because a ladder is not a scalar).
    """
    from .ranking import _matching_stems, load_results

    results = load_results(model_key)
    stems = list(results) if not files else [
        s for spec in files for s in _matching_stems(spec, results)
    ]
    for stem in stems:
        for record in results[stem].get("results", []):
            if record.get("metric_id") == metric_id:
                return record
    return None
