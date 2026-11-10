"""Shared style for the documentation figures (Nature-like).

One canvas width for every figure (183 mm, Nature double column); heights are
free.  Fonts are set once here in PAPER_FONTS and read by every notebook
through ``use_paper_fonts()``; panel code never writes a number.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox

# ----------------------------------------------------------------- locations
HERE = Path(__file__).resolve().parent
RELEASE = HERE.parents[1]
FIG_ROOT = RELEASE / "docs" / "figures"
# Code: the in-repo eval_framework (or $MLIPEVAL_HOME).  Data: results/ and
# data/ are not in the repository -- the figures read them from wherever the site
# config puts them (paths.output / $MLIPEVAL_OUTPUT, see mlipeval.config).
EVAL_HOME = Path(os.environ.get("MLIPEVAL_HOME", RELEASE / "eval_framework")).resolve()
os.environ.setdefault("MLIPEVAL_MODEL", "mace_mpa0")
if str(EVAL_HOME / "scripts") not in sys.path:
    sys.path.insert(0, str(EVAL_HOME / "scripts"))

# ----------------------------------------------------------------- geometry
MM = 1 / 25.4
W2 = 183 * MM          # canvas width, inches — never varies between figures
W1 = 89 * MM           # single column, for reference only

# ----------------------------------------------------------------- fonts
PAPER_FONTS = {
    "title": 7, "label": 7, "tick": 6, "legend": 6, "annot": 6,
    "panel": 8, "cbar_label": 6, "cbar_tick": 5.5,
}


def use_paper_fonts() -> dict:
    """Apply the shared rcParams and return the font table."""
    fs = PAPER_FONTS
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
        "font.size": fs["label"],
        "axes.titlesize": fs["title"], "axes.labelsize": fs["label"],
        "xtick.labelsize": fs["tick"], "ytick.labelsize": fs["tick"],
        "legend.fontsize": fs["legend"],
        "axes.linewidth": 0.5, "axes.edgecolor": "#222222", "axes.labelcolor": "#222222",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlepad": 3.0, "axes.labelpad": 2.0, "axes.grid": False,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "xtick.major.size": 2.0, "ytick.major.size": 2.0,
        "xtick.minor.size": 1.2, "ytick.minor.size": 1.2,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
        "xtick.color": "#222222", "ytick.color": "#222222",
        "lines.linewidth": 0.9, "lines.markersize": 3.0,
        "patch.linewidth": 0.5,
        "legend.frameon": False, "legend.handlelength": 1.6, "legend.handletextpad": 0.4,
        "legend.borderaxespad": 0.2, "legend.columnspacing": 1.0, "legend.labelspacing": 0.25,
        "figure.dpi": 150, "savefig.dpi": 600,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    return dict(fs)


# ----------------------------------------------------------------- model identity
# key: (short label, colour, marker, group).  Universal potentials are cool hues
# with filled markers; specialised potentials warm hues, open markers in scatter.
MODELS = {
    "mace_mpa0":        ("MACE-MPA-0",       "#3C5488", "o", "universal"),
    "nequip_oam_l":     ("NequIP-OAM-L",     "#4DBBD5", "s", "universal"),
    "dpa4_pro":         ("DPA-4.0.1-Pro",    "#00A087", "^", "universal"),
    "mattersim_5m":     ("MatterSim-5M",     "#8491B4", "D", "universal"),
    "chgnet_030":       ("CHGNet-0.3.0",     "#B09C85", "v", "universal"),
    "mace_spec_gst":    ("MACE-spec-GST",    "#E64B35", "o", "specialised"),
    "mace_spec_gst_ne": ("MACE-spec-GST-NE", "#F39B7F", "P", "specialised"),
    "nequip_spec_gst":  ("NequIP-spec-GST",  "#DC9100", "s", "specialised"),
    "mace_gst_aging":   ("MACE-GST-Aging",   "#7E6148", "X", "specialised"),
}
UNIVERSAL = [k for k, v in MODELS.items() if v[3] == "universal"]
SPECIALISED = [k for k, v in MODELS.items() if v[3] == "specialised"]
SHOWCASE = UNIVERSAL + SPECIALISED
# Curves get crowded with nine lines: curve panels show two architecture pairs,
# universal vs specialised, so the only thing that differs is the training data.
FOCUS = ["mace_mpa0", "mace_spec_gst", "nequip_oam_l", "nequip_spec_gst"]
REF_COLOR = "#111111"
BAND_COLOR = "#D9D9D9"

AL_LADDER = ["al_gen0", "al_gen3", "al_gen4", "al_gen6", "al_gen7", "al_gen8", "al_gen9", "al_prod"]
AL_SHORT = {"al_gen0": "gen0", "al_gen3": "gen3", "al_gen4": "gen4", "al_gen6": "gen6",
            "al_gen7": "gen7", "al_gen8": "gen8", "al_gen9": "gen9", "al_prod": "prod"}
AL_FRAMES = {"al_gen0": 422, "al_gen3": 780, "al_gen4": 1060, "al_gen6": 1013,
             "al_gen7": 1161, "al_gen8": 701, "al_gen9": 1020, "al_prod": 3864}

LEVELS = ["L0", "L1", "L2", "L3", "L4", "L5"]
LEVEL_NAMES = {"L0": "Reference", "L1": "PES sanity", "L2": "Static", "L3": "Structure",
               "L4": "Dynamics", "L5": "Pathway"}


def lab(k: str) -> str:
    return MODELS[k][0]


def col(k: str) -> str:
    return MODELS[k][1]


def mkr(k: str) -> str:
    return MODELS[k][2]


# ----------------------------------------------------------------- data helpers
def results(model: str, stem_glob: str) -> dict:
    """{metric_id: record} for one model and one result-file glob."""
    from mlipeval import paths
    root = paths.MODEL_RESULTS_ROOT / model
    out = {}
    for p in sorted(root.glob(stem_glob)):
        for r in json.loads(p.read_text()).get("results", []):
            out[r["metric_id"]] = r
    return out


def npz(model: str, name: str):
    from mlipeval import paths
    p = paths.MODEL_DATA_ROOT / model / name
    if not p.exists():
        return None
    with np.load(p, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def ranking(models, clean: bool = True):
    from mlipeval.ranking import (INDICATORS, build_ranking, contaminated_metric_ids,
                                  gate_is_contaminated, load_results)
    exclude = frozenset(contaminated_metric_ids(models)) if clean else frozenset()
    ind = INDICATORS
    if clean and any(gate_is_contaminated(load_results(k)) for k in models):
        ind = tuple(i for i in INDICATORS if i.key != "L1 gate")
    return build_ranking(list(models), exclude_metrics=exclude, indicators=ind)


def gate_failed(model: str) -> list[str]:
    from mlipeval.ranking import load_results
    res = load_results(model)
    l1 = next((v for k, v in res.items() if k.startswith("L1")), None) or {}
    return list((l1.get("context") or {}).get("gate", {}).get("failed_metrics", []))


# ----------------------------------------------------------------- layout helpers
def mark(fig, ax, letter: str, fs: float, dx_in: float = 0.0, dy_in: float = 0.0):
    """Panel letter at the top-left of the axes' tight bounding box."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bb = ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
    W, H = fig.get_size_inches()
    x = max(bb.x0 - 0.02 / W + dx_in / W, 0.004)
    fig.text(x, bb.y1 + dy_in / H, letter, fontsize=fs,
             fontweight="bold", ha="left", va="bottom")


def save_matched_panels(fig, groups: dict, out_dir: Path, pad_in: float = 0.03) -> dict:
    """Crop each panel group out of the full canvas (same scale as the composite)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = {}
    for letter, g in groups.items():
        bbs = [a.get_tightbbox(r) for a in g.get("axes", [])]
        bbs += [t.get_window_extent(r) for t in g.get("texts", [])]
        bb = Bbox.union(bbs).transformed(fig.dpi_scale_trans.inverted()).padded(pad_in)
        boxes[letter] = bb
        fig.savefig(out_dir / f"{letter}.png", bbox_inches=bb, dpi=600, facecolor="white")
    return boxes


def export(fig, name: str):
    """Composite: 600-dpi PNG for print, 300-dpi PNG for the README, and a vector PDF."""
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_ROOT / f"{name}.pdf", facecolor="white")
    fig.savefig(FIG_ROOT / f"{name}.png", dpi=300, facecolor="white")
    return FIG_ROOT / f"{name}.png"


def legend_handles(keys, ls="-", marker=True):
    from matplotlib.lines import Line2D
    return [Line2D([], [], color=col(k), ls=ls, marker=mkr(k) if marker else None,
                   ms=3, mfc=col(k), mec="white", mew=0.3, label=lab(k)) for k in keys]


class Slots(dict):
    """letter -> SubplotSpec; iterating yields the specs in panel order (what figcheck expects)."""

    def __iter__(self):
        return iter(self.values())
