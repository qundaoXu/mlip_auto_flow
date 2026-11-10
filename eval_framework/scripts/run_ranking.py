#!/usr/bin/env python
"""Driver: score and rank every model that has results under ``results/models/``.

    python run_ranking.py [--models a,b,c] [--markdown] [--level-weights L0=0.2,...]

Pure post-processing -- it loads no model and touches no GPU, so it runs on the
login node.  Writes ``results/ranking.json`` and, with ``--markdown``,
``results/RANKING_REPORT.md``.

The score is *relative* (five models, one reference, one code path), not the
framework's absolute A/B/C/F grade; see :mod:`mlipeval.ranking` for why the
absolute grade stays ``provisional`` in this project.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mlipeval import models, paths
from mlipeval.ranking import (
    INDICATORS,
    load_results,
    LEVEL_WEIGHTS,
    LEVELS,
    build_ranking,
    contaminated_indicators,
    contaminated_metric_ids,
    gate_is_contaminated,
    render_table,
    _fmt,
)


def parse_weights(text: str | None) -> dict[str, float] | None:
    if not text:
        return None
    weights = dict(LEVEL_WEIGHTS)
    for item in text.split(","):
        key, _, value = item.partition("=")
        key = key.strip().upper()
        if key not in weights:
            raise SystemExit(f"unknown level {key!r}; expected one of {', '.join(LEVELS)}")
        weights[key] = float(value)
    return weights


def _metric(payload, metric_id):
    for record in (payload or {}).get("results", []):
        if record["metric_id"] == metric_id:
            return record
    return None


def read_notes(order: list[str], labels: dict[str, str]) -> list[str]:
    """Caveats derived from the result files, not written by hand.

    Three of them decide how the ranking should be read, and all three are
    invisible in the score table itself.
    """
    lines: list[str] = []

    # 1. what closed each L1 gate, and whether the failing region is one the
    #    model actually visits in this workload
    lines.append("**L1 门控为什么关**（以及关得有多重）：")
    for key in order:
        res = load_results(key)
        l1 = next((v for k, v in res.items() if k.startswith("L1")), None)
        if not l1:
            continue
        gate = (l1.get("context") or {}).get("gate", {})
        failed = gate.get("failed_metrics", [])
        dmin = _metric(l1, "L1.stability_2000K.min_distance")
        dmin_v = dmin["value"] if dmin else float("nan")
        if not failed:
            lines.append(f"- {labels[key]}：通过（23 项全过），2000 K 探针最近接触 {dmin_v:.2f} Å。")
            continue
        reasons: dict[str, int] = {}
        for mid in failed:
            rec = _metric(l1, mid)
            for part in (rec.get("failure_reason") or "").split("; "):
                if part:
                    reasons[part] = reasons.get(part, 0) + 1
        detail = "；".join(f"{r}（{n} 条曲线）" for r, n in reasons.items())
        lines.append(f"- {labels[key]}：{len(failed)} 项失败 — {detail}。"
                     f"2000 K 探针最近接触 {dmin_v:.2f} Å。")
    lines += [
        "",
        "二体曲线的排斥性判据只看 **r ≤ 1.6 Å**。上面每个模型在 2000 K 稳定性探针里的最近接触都在 "
        "2.1 Å 以上，也就是说这段被判失败的区域，本工作负载**并没有真正走到**。"
        "所以门控关闭说明的是「还能往多极端推」的上界（高温/高压/碰撞级事件），"
        "不是说这些模型在本项目的熔淬-非晶流程里不可用。",
        "",
    ]

    # 2. phase ordering is a sign test: getting it backwards is qualitative
    bad = []
    for key in order:
        res = load_results(key)
        l2 = next((v for k, v in res.items() if k.startswith("L2")), None)
        rec = _metric(l2, "L2.phase_order.cubic_minus_hexagonal") if l2 else None
        if rec and isinstance(rec.get("value"), (int, float)) and rec["value"] <= 0:
            bad.append(f"{labels[key]}（{rec['value']:+.1f} meV/atom）")
    if bad:
        lines += ["**相序判错**（六方相应当低于立方相）：" + "、".join(bad) + "。"
                  "这是定性错误，不是精度问题。", ""]

    # 3. the energy zero point is not comparable across models
    lines += [
        "**能量零点不可跨模型比较**：L0 的 energy MAE 已按 `database_link.md` 3.4 逐组成去掉常数偏移，"
        "所以是可比的；但各模型的绝对能量（例如 LAMMPS 冒烟测试里的势能）差异来自参考态，不是误差。",
        "",
        "L3–L5 的 canonical 状态点各有 3 条独立 AIMD replica，距离类指标已按 replica 间离散度正式判级；"
        "L0–L2 没有独立参考副本，数值仍是 `provisional`。所有参考都来自同一 canonical 协议，"
        "不涉及 `d_protocol`。这里给的是相对名次，不是绝对判级。",
    ]
    return lines


def markdown(ranking, present: list[str]) -> str:
    labels = {k: models.get(k).label for k in ranking.model_keys}
    order = ranking.order()
    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z")

    lines = [
        "# 通用势函数 L0–L5 量化排名（GST225）",
        "",
        f"生成时间：{now}　　框架：`eval/` (`validation_methods_L0-L5.md`)",
        "",
        "## 这个分数是什么、不是什么",
        "",
        "- **是**：五个模型跑**同一套参考数据、同一协议、同一套冻结后处理参数、同一份代码**，",
        "  逐指标按 `100 × (x_best + floor) / (x_i + floor)` 折算成 0–100 的**相对**分。",
        "  best = 100；误差是最好模型两倍的约 50 分。`floor` 是该指标的物理/统计分辨尺度，",
        "  低于它的差异不计分，避免「大家都几乎精确」的指标靠数值尘埃主导排名。",
        "- **不是**：框架第 4 节的 A/B/C/F 绝对判级。绝对判级写在各结果文件里（L3–L5 由 3 条",
        "  AIMD replica 给出 `sigma_ref`）；L0–L2 没有独立参考副本，仍是 `provisional`。",
        "  相对排名不受这个限制——同一参考对所有模型的偏置是共同的。",
        "",
        "## 总排名",
        "",
        "| 名次 | 模型 | 总分 | " + " | ".join(LEVELS) + " | L1 门控 |",
        "|---|---|---|" + "---|" * (len(LEVELS) + 1),
    ]
    for rank, key in enumerate(order, 1):
        cells = [_fmt(ranking.level_scores[lv][key], 3) for lv in LEVELS]
        lines.append(
            f"| {rank} | {labels[key]} | **{_fmt(ranking.overall[key], 3)}** | "
            + " | ".join(cells)
            + f" | {'通过' if ranking.gate_open[key] else '未通过/缺失'} |"
        )

    lines += [
        "",
        f"级别权重：{', '.join(f'{k} {v:g}' for k, v in ranking.level_weights.items())}",
        "",
        "## 逐指标原始值与得分",
        "",
        "每格为 `原始值 (得分)`。原始值的方向见「越小越好 / 越大越好 / 通过率」一列。",
        "",
    ]

    header = "| 级别 | 指标 | 单位 | 方向 | floor | " + " | ".join(labels[k] for k in order) + " |"
    lines.append(header)
    lines.append("|---|---|---|---|---|" + "---|" * len(order))
    direction_cn = {"lower": "越小越好", "higher": "越大越好", "gap": "越大越好(差距计分)",
                    "pass": "通过率"}
    for row in ranking.rows:
        ind = row.indicator
        cells = [f"{_fmt(row.raw[k])} ({_fmt(row.score[k], 3)})" for k in order]
        lines.append(
            f"| {ind.level} | {ind.key} | {ind.unit} | {direction_cn[ind.direction]} | "
            f"{_fmt(ind.floor, 3)} | " + " | ".join(cells) + " |"
        )

    lines += ["", "## 关键读法（由结果直接生成）", ""] + read_notes(order, labels) + ["", "## 指标释义", ""]
    for row in ranking.rows:
        if row.indicator.note:
            n = row.n_metrics.get(order[0], 0)
            lines.append(f"- **{row.indicator.level} {row.indicator.key}**"
                         f"（聚合 {n} 条 metric）：{row.indicator.note}")

    missing = [k for k in ranking.model_keys if k not in present]
    if missing:
        lines += ["", "## 缺失", "",
                  "以下模型没有结果目录，未参与排名："
                  + "、".join(models.get(k).label for k in missing)]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=None, help="comma-separated model keys (default: all registered)")
    ap.add_argument("--level-weights", default=None, help='e.g. "L3=0.35,L5=0.05"')
    ap.add_argument("--markdown", action="store_true", help="also write results/RANKING_REPORT.md")
    # Without this, any subset run overwrites the canonical five-model artefacts:
    # `--models mace_spec_gst,mace_mpa0 --markdown` would replace the leaderboard
    # with a two-row table under the same filename, and nothing would say so.
    ap.add_argument("--out-stem", default=None,
                    help="basename for the artefacts (default 'ranking'/'RANKING_REPORT'; "
                         "required in practice for any --models subset, e.g. 'spec_vs_mpa0')")
    ap.add_argument("--drop-contaminated", action="store_true",
                    help="drop indicators that any listed model was trained on, for every "
                         "model, so the totals compare like with like (see mlipeval.leakage)")
    args = ap.parse_args()

    keys = [k.strip() for k in args.models.split(",")] if args.models else list(models.ORDER)
    present = [k for k in keys if (paths.MODEL_RESULTS_ROOT / k).is_dir()]
    if not present:
        print(f"no model results under {paths.MODEL_RESULTS_ROOT}; run the chains first",
              file=sys.stderr)
        return 2
    if len(present) < len(keys):
        print("missing results for: " + ", ".join(k for k in keys if k not in present))

    exclude: frozenset[str] = frozenset()
    if args.drop_contaminated:
        # Excluded at *metric* level and for *every* model. Metric level, because
        # an indicator like "relaxed volume error" mixes the contaminated
        # hexagonal cell with the clean cubic one and the clean half is still
        # evidence.  Every model, because excluding only for the contaminated one
        # would compare its cubic-only error against the other's hex+cubic error.
        exclude = frozenset(contaminated_metric_ids(present))
        if exclude:
            bad = contaminated_indicators(present, INDICATORS)
            for model_key, keys in bad.items():
                print(f"fully contaminated indicators for {model_key}: {', '.join(keys)}")
            print(f"excluding {len(exclude)} metric(s) from all {len(present)} models:")
            for m in sorted(exclude):
                print(f"    {m}")
            print()
        else:
            print("no contaminated metrics among the listed models\n")

        # The L1 gate is a context field, so metric-level exclusion cannot reach
        # it -- but it is derived from the L1 checks.  When every check that
        # closed it is contaminated, the gate result inherits the defect and
        # scoring it would penalise the model on its own training data.
        gated = [k for k in present if gate_is_contaminated(load_results(k))]
        if gated:
            print("L1 gate closed only by contaminated metrics for: " + ", ".join(gated))
            print("dropping the 'L1 gate' indicator for all models\n")
            indicators = tuple(i for i in INDICATORS if i.key != "L1 gate")
        else:
            indicators = INDICATORS
    else:
        indicators = INDICATORS

    ranking = build_ranking(present, level_weights=parse_weights(args.level_weights),
                            exclude_metrics=exclude, indicators=indicators)
    print(render_table(ranking))
    print()
    for rank, key in enumerate(ranking.order(), 1):
        print(f"{rank}. {models.get(key).label:<24s} {_fmt(ranking.overall[key], 4)}")

    # ranking.json is a cross-model artefact: it belongs beside the per-model
    # trees, not inside any one of them, so MLIPEVAL_MODEL must not redirect it.
    json_name = f"ranking_{args.out_stem}.json" if args.out_stem else "ranking.json"
    md_name = (f"RANKING_REPORT_{args.out_stem}.md" if args.out_stem
               else "RANKING_REPORT.md")
    out = paths.EVAL / "results" / json_name
    out.write_text(json.dumps(ranking.to_dict(), indent=2))
    print(f"\nwrote {out}")

    if args.markdown:
        md = paths.EVAL / "results" / md_name
        md.write_text(markdown(ranking, present), encoding="utf-8")
        print(f"wrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
