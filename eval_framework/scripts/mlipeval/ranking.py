"""Cross-model scoring: turn five models' L0-L5 result files into one ranking.

**What this is not.** ``validation_methods_L0-L5.md`` section 4 grades a model
against the *reference* with ``sigma_eff = max(sigma_ref, d_protocol/2)``.  That
absolute grade lives in the result files, not here: since 2026-10-27 the L3-L5
state points have three AIMD replicas and :mod:`mlipeval.ensemble` grades their
distance metrics against the replica spread.  L0 is scored on the H5
``protected_audit`` relabels, which share the canonical protocol with every
other reference, so no ``d_protocol`` enters; its metrics still carry no
replica spread and stay ``provisional``.

**What this is.** A *relative* comparison: five models evaluated against the
same reference, with the same protocol, the same frozen post-processing
parameters and the same code path.  Their errors are therefore directly
comparable to each other even though none of them can be graded in absolute
terms.  Each indicator is scored

    score_i = 100 * (x_best + floor) / (x_i + floor)

so the best model scores 100 and a model with twice the error of the best
scores ~50.  ``floor`` is the scale below which a difference stops being
physically meaningful -- the measured statistical noise where the framework
provides one (:mod:`mlipeval.uncertainty`), otherwise an explicit physical
resolution written into :data:`INDICATORS`.  Without a floor an indicator on
which every model is essentially exact would dominate the ranking through
numerical dust.

Boolean sanity checks (L1) score 100 or 0.  The L1 gate is carried as the
heaviest single indicator *and* reported as its own column, but it does not
truncate the overall score: framework rule 6 is about not *spending* compute on
L4/L5 behind a closed gate, and by the time this runs that compute is already
spent.  What a closed gate does mean is that the L2-L5 numbers of that model are
advisory -- read the gate column before the ranking.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from . import models as model_registry
from .paths import EVAL, MODEL_RESULTS_ROOT

# --------------------------------------------------------------------------
# indicator definitions
# --------------------------------------------------------------------------
Aggregator = Callable[[Sequence[float]], float]

#: Result files of the active (canonical, three-replica) state points.  Since
#: 2026-10-27 L3-L5 are scored on these; the ``canon_`` prefix keeps prefix
#: matching away from the legacy files archived on 2026-11-02.
L3_STATES = ("canon_amorphous_300K", "canon_liquid_950K", "canon_liquid_1200K", "canon_liquid_3000K")
L3A_FILES = tuple(f"L3a_{s}" for s in L3_STATES)
L3B_FILES = tuple(f"L3b_{s}" for s in L3_STATES)
L4_FILES = ("L4_canon_liquid_950K", "L4_canon_liquid_1200K", "L4_canon_liquid_3000K")
L5_FILES = ("L5_canon_quench", "L5_canon_heating")


def _mean_abs(values: Sequence[float]) -> float:
    return float(np.mean(np.abs(values)))


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values))


def _max_abs(values: Sequence[float]) -> float:
    return float(np.max(np.abs(values)))


def _min(values: Sequence[float]) -> float:
    return float(np.min(values))


@dataclass(frozen=True)
class Indicator:
    """One scored quantity, and everything needed to extract and interpret it."""

    key: str
    level: str
    files: tuple[str, ...]  # result-file stems to search, in order
    pattern: str  # regex matched against metric_id
    unit: str
    floor: float  # difference below this is physically meaningless
    weight: float = 1.0
    #: ``lower``   -- ratio score on |x|, best = 100 (the default)
    #: ``higher``  -- ratio score on x, best = 100 (bounded fractions, rates)
    #: ``gap``     -- ``100 * floor / (best - x + floor)``: for safety margins,
    #:               where being *near* the best is what matters, not the ratio
    #: ``pass``    -- x is already a 0-1 pass fraction
    direction: str = "lower"
    #: ``one_minus`` turns a goodness-of-fit like R^2 into the error 1 - R^2, so
    #: 0.97 vs 0.71 is scored as 0.03 vs 0.29 rather than as a 25% difference
    transform: str | None = None
    field: str = "value"  # "value" | "status_ok" | "extra.<key>" | "sign_positive"
    aggregate: Aggregator = _mean_abs
    note: str = ""


#: The scored set.  Deliberately small: every entry has to be defensible as
#: "this is what makes a GST225 potential usable", not merely "this number exists".
INDICATORS: tuple[Indicator, ...] = (
    # ---- L0: static accuracy against the H5 protected-audit relabels -------
    Indicator(
        key="force MAE", level="L0", files=("L0_audit",),
        pattern=r"^L0\.force_mae$", unit="eV/A", floor=0.02, weight=2.0,
        note="the offset-free accuracy statement; forces need no energy reference",
    ),
    Indicator(
        key="energy MAE", level="L0", files=("L0_audit",),
        pattern=r"^L0\.energy_mae$", unit="meV/atom", floor=5.0, weight=1.5,
        note="after per-composition constant offset removal (database_link.md 3.4)",
    ),
    Indicator(
        key="stress RMSE", level="L0", files=("L0_audit",),
        pattern=r"^L0\.stress_rmse$", unit="meV/A^3", floor=0.1, weight=1.0,
        note="tier B: REF_stress read from the canonical static OUTCARs, no sign to audit",
    ),
    Indicator(
        key="force 1-R2", level="L0", files=("L0_audit",),
        pattern=r"^L0\.force_component_r2$", unit="-", floor=0.01, weight=1.0,
        transform="one_minus",
        note="scored as the unexplained variance 1 - R^2, not as R^2 itself",
    ),
    Indicator(
        key="force MAE worst stage", level="L0", files=("L0_audit",),
        pattern=r"^L0\.force_rmse\.pool=", unit="eV/A", floor=0.02, weight=0.5,
        aggregate=_max_abs,
        note=("worst of the six H5 pools (950 K, heating, 3000 K melt, 1200 K, quench, "
              "300 K glass): transferability across state kinds"),
    ),

    # ---- L1: PES sanity and trajectory stability (reference-free) ---------
    Indicator(
        key="L1 gate", level="L1", files=("L1",),
        pattern=r"^context:gate\.pass$", unit="-", floor=0.0, weight=2.0,
        direction="pass",
        note="framework rule 6: L1 failure blocks L2-L5.  Binary, so it says nothing "
             "about *how* badly a model missed -- read it with the two PEC rows below",
    ),
    Indicator(
        key="PEC curves passing L1", level="L1", files=("L1",),
        pattern=r"^L1\.pec\.", unit="-", floor=0.0, weight=1.5,
        direction="pass", field="status_ok", aggregate=_mean,
        note="fraction of the six binary curves that pass every L1 continuity check; "
             "finer-grained than the binary gate",
    ),
    Indicator(
        key="NVE drift", level="L1", files=("L1",),
        pattern=r"^L1\.nve_drift\.dt=3fs$", unit="meV/atom/ps", floor=0.05, weight=2.0,
        note="tier A: same cell, same dt, same programme as the AIMD preflight",
    ),
    Indicator(
        key="energy-force consistency", level="L1", files=("L1",),
        pattern=r"^L1\.(force_vs_dE_dx|stress_vs_dE_dstrain)$", unit="-",
        floor=5e-3, weight=1.0,
        note=("analytic derivative vs finite difference; the floor is the framework's "
              "own 5e-3 pass threshold, below which tighter agreement buys nothing"),
    ),
    Indicator(
        key="PEC repulsive walls", level="L1", files=("L1",),
        pattern=r"^L1\.pec\.", unit="-", floor=0.0, weight=1.5,
        direction="pass", field="extra.monotonic_repulsive", aggregate=_mean,
        note="fraction of the six binary curves that rise monotonically into the core",
    ),
    Indicator(
        key="2000 K min distance", level="L1", files=("L1",),
        pattern=r"^L1\.stability_2000K\.min_distance$", unit="A", floor=0.05, weight=1.0,
        direction="gap",
        note="a collapsing model produces sub-2 A contacts; higher is safer",
    ),
    Indicator(
        key="lattice scan smoothness", level="L1", files=("L1",),
        pattern=r"^L1\.lattice_scan_smoothness$", unit="max/median |d2E|", floor=1.0,
        weight=0.5,
    ),
    Indicator(
        key="timestep drift spread", level="L1", files=("L1",),
        pattern=r"^L1\.timestep_sensitivity\.drift_spread$", unit="meV/atom/ps",
        floor=0.05, weight=1.0,
        note="how much the NVE drift changes between dt = 3.0 and 1.5 fs",
    ),

    # ---- L2: static equilibrium of the ordered phases ---------------------
    Indicator(
        key="relaxed volume error", level="L2", files=("L2",),
        pattern=r"^L2\.relax\..*\.volume_error$", unit="%", floor=0.5, weight=2.0,
    ),
    Indicator(
        key="relaxed cell-length error", level="L2", files=("L2",),
        pattern=r"^L2\.relax\..*\.cell_length_error$", unit="%", floor=0.5, weight=1.0,
    ),
    Indicator(
        key="EOS V0 error", level="L2", files=("L2",),
        pattern=r"^L2\.eos\..*\.V0_error$", unit="%", floor=0.5, weight=1.5,
    ),
    Indicator(
        key="EOS B0 error", level="L2", files=("L2",),
        pattern=r"^L2\.eos\..*\.B0_error$", unit="%", floor=5.0, weight=1.5,
        note="bulk modulus is the curvature of the EOS: far more sensitive than V0",
    ),
    Indicator(
        key="phase order", level="L2", files=("L2",),
        pattern=r"^L2\.phase_order\.cubic_minus_hexagonal$", unit="meV/atom",
        floor=0.0, weight=1.0, direction="pass", field="sign_positive",
        note="hexagonal must come out below cubic; this scores the sign, not the size",
    ),
    Indicator(
        key="phase ladder ordering", level="L2", files=("L2",),
        pattern=r"^L2\.phase_ladder\.inversions$", unit="count",
        floor=0.0, weight=1.5, direction="pass", field="status_ok",
        note=(
            "all six pairs of the four-rung DFT ladder (two P-3m1 stackings, "
            "monoclinic Cm, cubic) in the right order.  The binary above only ever "
            "tested one pair; a model can get the extremes right and scramble the middle"
        ),
    ),
    Indicator(
        key="phase ladder dE error", level="L2", files=("L2",),
        pattern=r"^L2\.phase_ladder\.dE_error\.", unit="meV/atom",
        floor=2.0, weight=2.0, direction="lower",   # _mean_abs already takes |.|
        note=(
            "per-rung energy error against the 54.7 meV/atom DFT ladder.  floor 2 "
            "meV/atom: below that the rungs are inside the protocol's own resolution"
        ),
    ),
    Indicator(
        key="stacking gap error", level="L2", files=("L2",),
        pattern=r"^L2\.phase_ladder\.dE_error\.mp3534$", unit="meV/atom",
        floor=2.0, weight=1.5, direction="lower",
        note=(
            "the two P-3m1 stackings are 15.13 meV/atom apart in DFT -- the sharpest "
            "test in the ladder, and the resolution a crystallisation study needs"
        ),
    ),

    Indicator(
        key="formation energy error", level="L2", files=("L2",),
        pattern=r"^L2\.formation_energy\.dE_error\.", unit="meV/atom",
        floor=2.0, weight=1.5, direction="lower",
        note=(
            "every ladder phase referenced to the L2 element ground states; per-element "
            "offsets of the model's native protocol cancel, so this is comparable across models"
        ),
    ),
    Indicator(
        key="corrected phase order", level="L2", files=("L2",),
        pattern=r"^L2\.formation_energy\.cubic_minus_hexagonal$", unit="meV/atom",
        floor=0.0, weight=1.0, direction="pass", field="extra.ordering_reproduced",
        note="cubic above hexagonal after the Ge43Sb43Te108 composition correction",
    ),

    # ---- L3: disordered structure, all isothermal state points ------------
    Indicator(
        key="partial RDF W1", level="L3", files=L3A_FILES,
        pattern=r"^L3a\.rdf\..*\.wasserstein1@", unit="A", floor=0.02, weight=2.0,
        note="all six partials, all four isothermal state points: the bond-length statement",
    ),
    Indicator(
        key="ADF JS", level="L3", files=L3A_FILES,
        pattern=r"^L3a\.adf\..*\.jensen_shannon@", unit="-", floor=0.02, weight=1.5,
    ),
    Indicator(
        key="CN mean error", level="L3", files=L3A_FILES,
        pattern=r"^L3a\.cn\..*\.mean@", unit="-", floor=0.05, weight=2.0,
        field="extra.difference",
    ),
    Indicator(
        key="homopolar bond error", level="L3", files=L3A_FILES,
        pattern=r"^L3a\.homopolar\.", unit="fraction of bonds", floor=0.004, weight=2.0,
        field="extra.difference",
        note="the chemical-order signature: MACE's headline phase-1 failure mode",
    ),
    Indicator(
        key="tetrahedral q W1", level="L3", files=("L3a_canon_amorphous_300K",),
        pattern=r"^L3a\.tetrahedral_q\..*\.wasserstein1@", unit="-", floor=0.02, weight=1.0,
    ),
    Indicator(
        key="ring distribution error", level="L3",
        files=L3B_FILES,
        pattern=r"^L3b\.rings\.size\d+@", unit="rings/atom", floor=0.02, weight=1.5,
        field="extra.difference",
        note="medium-range topology; the 4-ring population is the cubic-order proxy",
    ),
    Indicator(
        key="ALTBC asymmetry error", level="L3",
        files=L3B_FILES,
        pattern=r"^L3b\.altbc\.asymmetry@", unit="A", floor=0.02, weight=1.0,
        field="extra.difference",
        note="Peierls-like long/short bond alternation, the PCM-specific observable",
    ),
    Indicator(
        key="principal peak error", level="L3",
        files=L3B_FILES,
        pattern=r"^L3b\.principal_peak\.q_pp@", unit="1/A", floor=0.02, weight=0.5,
        field="extra.difference",
    ),

    # ---- L4: dynamics -----------------------------------------------------
    Indicator(
        key="diffusion dlog10D", level="L4", files=L4_FILES,
        pattern=r"^L4\.diffusion\..*\.delta_log10D@", unit="decades", floor=0.05,
        weight=3.0,
        note="log-space, so it is symmetric in over- and under-estimating D",
    ),
    Indicator(
        key="equilibrium segment", level="L4", files=L4_FILES,
        pattern=r"^L4\.equilibrium_segment@.*\.model$", unit="-", floor=0.0,
        weight=1.0, direction="pass",
        note="the model MD must itself be an equilibrium segment before D means anything",
    ),

    # ---- L5: quench and heating pathways --------------------------------
    Indicator(
        key="pathway endpoint error", level="L5", files=L5_FILES,
        pattern=r"^L5\.pathway\..*\.final_state_error@", unit="-", floor=0.02,
        weight=2.0,
        note="relative error of each observable at the end of the quench and heating programmes",
    ),
    Indicator(
        key="first divergence", level="L5", files=L5_FILES,
        pattern=r"^L5\.pathway\..*\.first_divergence_progress@", unit="fraction",
        floor=0.05, weight=1.0, direction="higher", aggregate=_mean,
        note="how far along the programme the model stays on the reference path",
    ),
)

#: Level weights for the overall score.  L3 carries the most because the phase-1
#: target is amorphous GST225 structure; L1 is weighted as a gate rather than as
#: a quality statement.  Override with ``--level-weights`` on the driver.
LEVEL_WEIGHTS = {"L0": 0.20, "L1": 0.15, "L2": 0.15, "L3": 0.25, "L4": 0.15, "L5": 0.10}

LEVELS = ("L0", "L1", "L2", "L3", "L4", "L5")


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------
def _dig(obj: Any, dotted: str) -> Any:
    for part in dotted.split("."):
        if obj is None:
            return None
        obj = obj.get(part) if isinstance(obj, dict) else None
    return obj


def _matching_stems(spec: str, stems: Iterable[str]) -> list[str]:
    """File specs are prefixes: ``"L1"`` matches ``L1_mace_mpa0.json``.

    Two drivers still name their output after the phase-1 model
    (``L1_mace_mpa0.json``, ``L2_mace_mpa0.json``); prefix matching keeps the
    indicator table readable instead of enumerating every model's filename.
    """
    return [s for s in stems if s == spec or s.startswith(spec + "_")]


def load_results(model_key: str) -> dict[str, dict]:
    """Every ``*.json`` result file of one model, keyed by file stem."""
    root = MODEL_RESULTS_ROOT / model_key
    out: dict[str, dict] = {}
    if not root.is_dir():
        return out
    from .state_points import MD_ENGINES, current_engine

    engine = current_engine()
    for path in sorted(root.glob("*.json")):
        stem = path.stem
        # 依赖 MD 的结果带 ``__<engine>`` 后缀。**只收当前引擎的**，
        # 并把后缀剥掉再入索引，这样下面的指标定义不必知道引擎的存在。
        # 不过滤的话，一次排名会把两个引擎的 L3/L4/L5 混在一起——
        # 那正好废掉"唯一变量是模型"这个前提。
        base, sep, tag = stem.rpartition("__")
        if sep and tag in MD_ENGINES:
            if tag != engine:
                continue
            stem = base
        try:
            out[stem] = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
    return out


def _field_value(record: dict, field_name: str) -> float | None:
    if field_name == "status_ok":
        # the framework's own verdict on this metric, as a 0/1 pass
        return float(record.get("status") == "ok")
    if field_name == "value":
        val = record.get("value")
    elif field_name == "sign_positive":
        val = record.get("value")
        return None if val is None else float(float(val) > 0)
    elif field_name.startswith("extra."):
        extra = record.get("extra") or {}
        val = _dig(extra, field_name[len("extra."):])
        if val is None and field_name == "extra.difference":
            # some L3b metrics record reference and model but not their difference
            ref, mod = extra.get("reference"), extra.get("model")
            if isinstance(ref, (int, float)) and isinstance(mod, (int, float)):
                val = float(mod) - float(ref)
    else:
        val = _dig(record, field_name)
    if isinstance(val, bool):
        return float(val)
    if isinstance(val, (int, float)) and math.isfinite(float(val)):
        return float(val)
    return None


def indicator_is_contaminated(indicator: Indicator, results: dict[str, dict]) -> bool:
    """Does this indicator draw on metrics the model was trained on?

    Decided from the data rather than by matching one regex against another: a
    metric carries ``extra.training_leakage`` only if
    :mod:`mlipeval.leakage` marked it at save time, so this asks the result file
    directly.  ``True`` means every value feeding the indicator would be a
    readout of training convergence.

    This matters because two L1 indicators ("PEC curves passing L1", "PEC
    repulsive walls", weight 1.5 each) aggregate ``^L1\\.pec\\.`` -- for a
    potential trained on dimers, scoring them would put 3.0 of L1's weight on
    training data.
    """
    rx = re.compile(indicator.pattern)
    if indicator.pattern.startswith("^context:"):
        return False
    stems: list[str] = []
    for spec in indicator.files:
        stems.extend(s for s in _matching_stems(spec, results) if s not in stems)
    matched = flagged = 0
    for stem in stems:
        for record in (results.get(stem) or {}).get("results", []):
            if not rx.search(record.get("metric_id", "")):
                continue
            matched += 1
            if "training_leakage" in (record.get("extra") or {}):
                flagged += 1
    return matched > 0 and flagged == matched


def contaminated_indicators(
    model_keys: Sequence[str], indicators: Iterable[Indicator] = None  # type: ignore[assignment]
) -> dict[str, list[str]]:
    """Indicator keys that are *entirely* contaminated, per model that has any."""
    indicators = INDICATORS if indicators is None else indicators
    out: dict[str, list[str]] = {}
    for key in model_keys:
        results = load_results(key)
        hits = [i.key for i in indicators if indicator_is_contaminated(i, results)]
        if hits:
            out[key] = hits
    return out


def gate_is_contaminated(results: dict[str, dict]) -> bool:
    """Was the L1 gate closed only by metrics the model was trained on?

    The gate is a context field, so no metric-level exclusion can reach it -- but
    it is *derived* from the L1 checks, and if every check that closed it is
    contaminated then the gate outcome carries exactly the same defect.  For
    ``mace_spec_gst`` all four failures are PEC curves it was trained on, so
    scoring the gate at 0 would penalise it on training data.
    """
    for stem, payload in results.items():
        if not stem.startswith("L1"):
            continue
        gate = _dig(payload.get("context") or {}, "gate") or {}
        failed = gate.get("failed_metrics") or []
        if not failed:
            continue
        by_id = {r.get("metric_id"): r for r in payload.get("results", [])}
        if all("training_leakage" in (by_id.get(m, {}).get("extra") or {}) for m in failed):
            return True
    return False


def contaminated_metric_ids(model_keys: Sequence[str]) -> set[str]:
    """Every metric id that any listed model was trained on.

    Returned as one set, to be excluded from *all* models rather than only from
    the contaminated one.  Excluding per-model would silently change what an
    indicator means between columns: ``relaxed volume error`` averages the
    hexagonal and cubic cells, and dropping only the hexagonal one for the
    specialised potential would compare its cubic-only error against the other
    model's hex+cubic error and print the difference as a result.
    """
    ids: set[str] = set()
    for key in model_keys:
        for payload in load_results(key).values():
            for record in payload.get("results", []):
                if "training_leakage" in (record.get("extra") or {}):
                    ids.add(record.get("metric_id", ""))
    return ids - {""}


def extract(
    indicator: Indicator,
    results: dict[str, dict],
    *,
    exclude_metrics: frozenset[str] | set[str] = frozenset(),
) -> tuple[float | None, int]:
    """Aggregate one indicator over every matching metric in every listed file."""
    rx = re.compile(indicator.pattern)
    values: list[float] = []

    stems: list[str] = []
    for spec in indicator.files:
        stems.extend(s for s in _matching_stems(spec, results) if s not in stems)

    if indicator.pattern.startswith("^context:"):
        dotted = indicator.pattern[len("^context:"):].rstrip("$").replace("\\", "")
        for stem in stems:
            ctx = (results.get(stem) or {}).get("context")
            val = _dig(ctx or {}, dotted)
            if isinstance(val, bool):
                values.append(float(val))
            elif isinstance(val, (int, float)):
                values.append(float(val))
        return (indicator.aggregate(values) if values else None), len(values)

    for stem in stems:
        payload = results.get(stem)
        if payload is None:
            continue
        for record in payload.get("results", []):
            metric_id = record.get("metric_id", "")
            if not rx.search(metric_id) or metric_id in exclude_metrics:
                continue
            val = _field_value(record, indicator.field)
            if val is not None:
                values.append(val)
    if not values:
        return None, 0
    if indicator.transform == "one_minus":
        values = [1.0 - v for v in values]
    return indicator.aggregate(values), len(values)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
@dataclass
class IndicatorRow:
    indicator: Indicator
    raw: dict[str, float | None] = field(default_factory=dict)
    n_metrics: dict[str, int] = field(default_factory=dict)
    score: dict[str, float | None] = field(default_factory=dict)


def score_indicator(indicator: Indicator, raw: dict[str, float | None]) -> dict[str, float | None]:
    """Score one indicator across models, with the indicator's own floor."""
    present = {k: v for k, v in raw.items() if v is not None}
    if not present:
        return {k: None for k in raw}

    if indicator.direction == "pass":
        # already a 0-1 fraction (booleans and pass rates alike)
        return {k: (100.0 * float(np.clip(v, 0.0, 1.0)) if v is not None else None)
                for k, v in raw.items()}

    floor = max(indicator.floor, 1e-12)

    if indicator.direction == "gap":
        # safety margins: what matters is how far below the best you are, not by
        # what factor -- a 2.18 A closest approach is not "half as good" as 2.19
        best = max(present.values())
        return {k: (100.0 * floor / ((best - present[k]) + floor) if k in present else None)
                for k in raw}

    if indicator.direction == "higher":
        best = max(present.values())
        return {k: (100.0 * (present[k] + floor) / (best + floor) if k in present else None)
                for k in raw}

    magnitudes = {k: abs(v) for k, v in present.items()}
    best = min(magnitudes.values())
    return {k: (100.0 * (best + floor) / (magnitudes[k] + floor) if k in magnitudes else None)
            for k in raw}


@dataclass
class Ranking:
    model_keys: list[str]
    rows: list[IndicatorRow]
    level_scores: dict[str, dict[str, float | None]]
    level_coverage: dict[str, dict[str, int]]
    overall: dict[str, float | None]
    gate_open: dict[str, bool]
    level_weights: dict[str, float]

    def order(self) -> list[str]:
        return sorted(
            self.model_keys,
            key=lambda k: (-(self.overall.get(k) or -1.0), k),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": {
                "score": "100 * (x_best + floor) / (x_i + floor) per indicator",
                "note": (
                    "relative ranking across models on a shared reference; NOT the "
                    "framework's absolute A/B/C/F grade, which needs sigma_ref and "
                    "d_protocol and stays provisional"
                ),
                "level_weights": self.level_weights,
            },
            "models": self.model_keys,
            "gate_open": self.gate_open,
            "level_scores": self.level_scores,
            "level_coverage": self.level_coverage,
            "overall": self.overall,
            "indicators": [
                {
                    "key": r.indicator.key,
                    "level": r.indicator.level,
                    "unit": r.indicator.unit,
                    "direction": r.indicator.direction,
                    "weight": r.indicator.weight,
                    "floor": r.indicator.floor,
                    "note": r.indicator.note,
                    "raw": r.raw,
                    "n_metrics": r.n_metrics,
                    "score": r.score,
                }
                for r in self.rows
            ],
        }


def build_ranking(
    model_keys: Sequence[str] | None = None,
    *,
    level_weights: dict[str, float] | None = None,
    indicators: Iterable[Indicator] = INDICATORS,
    exclude_metrics: frozenset[str] | set[str] = frozenset(),
) -> Ranking:
    keys = list(model_keys or model_registry.ORDER)
    weights = dict(level_weights or LEVEL_WEIGHTS)
    per_model = {k: load_results(k) for k in keys}

    rows: list[IndicatorRow] = []
    for indicator in indicators:
        row = IndicatorRow(indicator=indicator)
        for key in keys:
            value, n = extract(indicator, per_model[key], exclude_metrics=exclude_metrics)
            row.raw[key] = value
            row.n_metrics[key] = n
        row.score = score_indicator(indicator, row.raw)
        rows.append(row)

    gate_open = {}
    for key in keys:
        gate = None
        for stem, payload in per_model[key].items():
            if stem.startswith("L1"):
                gate = _dig(payload.get("context") or {}, "gate.pass")
        gate_open[key] = bool(gate) if gate is not None else False

    level_scores: dict[str, dict[str, float | None]] = {}
    level_coverage: dict[str, dict[str, int]] = {}
    for level in LEVELS:
        level_rows = [r for r in rows if r.indicator.level == level]
        level_scores[level] = {}
        level_coverage[level] = {}
        for key in keys:
            num = sum(r.indicator.weight * r.score[key] for r in level_rows if r.score[key] is not None)
            den = sum(r.indicator.weight for r in level_rows if r.score[key] is not None)
            level_scores[level][key] = (num / den) if den else None
            level_coverage[level][key] = sum(1 for r in level_rows if r.score[key] is not None)

    overall: dict[str, float | None] = {}
    for key in keys:
        num = sum(weights[lv] * level_scores[lv][key] for lv in LEVELS if level_scores[lv][key] is not None)
        den = sum(weights[lv] for lv in LEVELS if level_scores[lv][key] is not None)
        overall[key] = (num / den) if den else None

    return Ranking(
        model_keys=keys, rows=rows, level_scores=level_scores,
        level_coverage=level_coverage, overall=overall, gate_open=gate_open,
        level_weights=weights,
    )


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "--"
    if isinstance(value, float) and (abs(value) >= 1e4 or (value and abs(value) < 1e-3)):
        return f"{value:.{digits - 1}e}"
    return f"{value:.{digits}g}"


def render_table(ranking: Ranking, labels: dict[str, str] | None = None) -> str:
    labels = labels or {k: model_registry.get(k).label for k in ranking.model_keys}
    order = ranking.order()
    head = ["level / indicator", "unit"] + [labels[k] for k in order]
    lines: list[list[str]] = [head]

    for level in LEVELS:
        level_rows = [r for r in ranking.rows if r.indicator.level == level]
        if not level_rows:
            continue
        lines.append([f"--- {level} ---", ""] + ["" for _ in order])
        for row in level_rows:
            lines.append(
                [f"  {row.indicator.key}", row.indicator.unit]
                + [f"{_fmt(row.raw[k])}  ({_fmt(row.score[k], 3)})" for k in order]
            )
        lines.append(
            [f"  = {level} score", "0-100"]
            + [_fmt(ranking.level_scores[level][k], 3) for k in order]
        )
    lines.append(["OVERALL", "0-100"] + [_fmt(ranking.overall[k], 3) for k in order])

    widths = [max(len(r[i]) for r in lines) for i in range(len(head))]
    out = []
    for i, row in enumerate(lines):
        out.append("  ".join(cell.ljust(widths[j]) for j, cell in enumerate(row)).rstrip())
        if i == 0:
            out.append("  ".join("-" * w for w in widths))
    return "\n".join(out)
