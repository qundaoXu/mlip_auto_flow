"""Shared plotting style and reusable panels.

Figures are returned, never written to disk: the notebooks display them inline
(phase-1 deliverable 2 explicitly does not want saved images).
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

SPECIES_COLORS = {"Ge": "#d1495b", "Sb": "#2e86ab", "Te": "#3f8f5b", "all": "#444444"}
PAIR_COLORS = {
    "Ge-Ge": "#d1495b",
    "Ge-Sb": "#e07a5f",
    "Ge-Te": "#f2a65a",
    "Sb-Sb": "#2e86ab",
    "Sb-Te": "#5fa8d3",
    "Te-Te": "#3f8f5b",
}
REF_STYLE = dict(color="#222222", lw=2.0, label="AIMD/DFT reference")
MODEL_STYLE = dict(color="#c1121f", lw=1.6, ls="--", label="MLIP")


def use_style() -> None:
    """Apply a compact, readable style. Call once per notebook."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 110,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 1.6,
            "figure.constrained_layout.use": True,
        }
    )


def species_color(name: str) -> str:
    return SPECIES_COLORS.get(name, "#777777")


def pair_color(key: str) -> str:
    return PAIR_COLORS.get(key, "#777777")


def parity_panel(ax, ref, model, *, unit: str, title: str = "", color: str = "#2e86ab", s: float = 6):
    """Parity scatter with a y=x guide and inline MAE/RMSE annotation."""
    ref = np.asarray(ref, dtype=float).reshape(-1)
    model = np.asarray(model, dtype=float).reshape(-1)
    ok = np.isfinite(ref) & np.isfinite(model)
    ref, model = ref[ok], model[ok]
    ax.scatter(ref, model, s=s, alpha=0.35, color=color, edgecolors="none", rasterized=True)
    lo = float(min(ref.min(), model.min()))
    hi = float(max(ref.max(), model.max()))
    pad = 0.03 * (hi - lo + 1e-12)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#222222", lw=1.0, ls=":")
    mae = float(np.mean(np.abs(model - ref)))
    rmse = float(np.sqrt(np.mean((model - ref) ** 2)))
    ax.set_xlabel(f"reference [{unit}]")
    ax.set_ylabel(f"model [{unit}]")
    ax.set_title(title)
    ax.text(
        0.03,
        0.97,
        f"MAE {mae:.4g}\nRMSE {rmse:.4g}\nN {ref.size}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbbbbb", alpha=0.85),
    )
    ax.set_aspect("equal", adjustable="datalim")
    return ax


def error_hist_panel(ax, errors, *, unit: str, title: str = "", color: str = "#c1121f", bins: int = 80):
    errors = np.asarray(errors, dtype=float).reshape(-1)
    errors = errors[np.isfinite(errors)]
    ax.hist(errors, bins=bins, color=color, alpha=0.75, edgecolor="none")
    for q, style in ((50, ":"), (99, "--")):
        v = np.percentile(np.abs(errors), q)
        ax.axvline(v, color="#222222", ls=style, lw=1.0)
        ax.axvline(-v, color="#222222", ls=style, lw=1.0)
    ax.set_xlabel(f"model - reference [{unit}]")
    ax.set_ylabel("count")
    ax.set_title(title)
    return ax


def binned_error_panel(ax, groups: Mapping[str, np.ndarray], *, unit: str, title: str = "", metric=np.abs):
    """Box-style summary of |error| per bin, with the count printed on top."""
    keys = list(groups)
    data = [np.asarray(groups[k], dtype=float) for k in keys]
    data = [d[np.isfinite(d)] for d in data]
    positions = np.arange(len(keys))
    ax.boxplot(
        [metric(d) if d.size else np.array([np.nan]) for d in data],
        positions=positions,
        widths=0.6,
        showfliers=False,
        medianprops=dict(color="#c1121f", lw=1.6),
    )
    for x, d in zip(positions, data):
        ax.text(x, ax.get_ylim()[1], f"n={d.size}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(positions)
    ax.set_xticklabels(keys, rotation=30, ha="right")
    ax.set_ylabel(f"|error| [{unit}]")
    ax.set_title(title)
    return ax


def compare_curves_panel(
    ax,
    x,
    ref_y,
    model_y=None,
    *,
    xlabel: str,
    ylabel: str,
    title: str = "",
    ref_label: str = "AIMD reference",
    model_label: str = "MLIP",
    color: str | None = None,
):
    ax.plot(x, ref_y, color=color or REF_STYLE["color"], lw=2.0, label=ref_label)
    if model_y is not None:
        ax.plot(x, model_y, color=MODEL_STYLE["color"], lw=1.6, ls="--", label=model_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return ax


def grade_badge(ax, result) -> None:
    """Stamp status/grade/tier of a MetricResult onto a panel."""
    tier = result.evidence_tier.value if result.evidence_tier else "-"
    text = f"{result.status.value} | tier {tier} | grade {result.grade.value}"
    colors = {"ok": "#3f8f5b", "provisional": "#e0a800", "failed": "#c1121f", "skipped": "#777777", "error": "#c1121f"}
    ax.text(
        0.99,
        0.02,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="white",
        bbox=dict(boxstyle="round,pad=0.3", fc=colors.get(result.status.value, "#777777"), ec="none", alpha=0.9),
    )


def results_table(results, *, max_rows: int = 60):
    """Styled dataframe of a ResultSet for notebook display."""
    import pandas as pd

    rows = results.summary_rows() if hasattr(results, "summary_rows") else [r.to_dict() for r in results]
    df = pd.DataFrame(rows).head(max_rows)
    return df
