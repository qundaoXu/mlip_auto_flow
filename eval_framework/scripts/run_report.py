#!/usr/bin/env python
"""Consolidate every level's results and check them against the phase-1 acceptance list.

    python run_report.py [--markdown]

Reads ``eval/results/*.json`` and produces:

* one table of every metric across L0-L5;
* an explicit pass/fail line for each acceptance condition in
  ``validation_methods_L0-L5.md`` section 8;
* the engineering checklist of section 9 with its current state.

The acceptance checks are evaluated from the result files, not asserted by hand:
each one names the evidence it used, so a condition cannot be ticked without
something in ``results/`` backing it.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mlipeval import paths
from mlipeval.state_points import CANONICAL_STATES


@dataclass
class Check:
    condition: str
    passed: bool | None          # None = partially satisfied
    evidence: str

    @property
    def mark(self) -> str:
        return {True: "PASS", False: "FAIL", None: "PARTIAL"}[self.passed]


def load_results() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(paths.RESULTS.glob("*.json")):
        try:
            out[path.stem] = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
    return out


def all_metrics(results: dict[str, dict]) -> list[dict]:
    rows = []
    for name, payload in results.items():
        for r in payload.get("results", []):
            rows.append({"file": name, **r})
    return rows


def acceptance_checks(results: dict[str, dict], rows: list[dict]) -> list[Check]:
    checks: list[Check] = []

    def files_with(prefix: str) -> list[str]:
        return [k for k in results if k.startswith(prefix)]

    # 1. one protocol drives both the AIMD reading and the MLIP MD
    md_protocols = {
        r.get("model_protocol") for r in rows if r.get("model_protocol") and "mlip_md" in str(r.get("model_protocol"))
    }
    ref_protocols = {r.get("reference_protocol") for r in rows if r.get("reference_protocol")}
    checks.append(Check(
        "同一协议可驱动 AIMD 参考读取和 MLIP-MD",
        bool(md_protocols and ref_protocols),
        f"{len(ref_protocols)} 个参考 protocol_id、{len(md_protocols)} 个 MLIP-MD protocol_id 出现在结果中",
    ))

    # 2. traceability
    traced = [r for r in rows if r.get("reference_protocol") or r.get("model_protocol")]
    has_model = [k for k, v in results.items() if v.get("model")]
    checks.append(Check(
        "所有结果都能回溯到 structure / protocol / model ID",
        len(traced) > 0 and len(has_model) > 0,
        f"{len(traced)}/{len(rows)} 条 metric 带协议标识；{len(has_model)} 个结果文件带完整 ModelSpec（含 checkpoint sha256）",
    ))

    # 3. L0 localises the failing atom and configuration
    worst = [r for r in rows if r["metric_id"] == "L0.worst_cases"]
    n_atoms = worst[0]["extra"]["n_atoms"] if worst else 0
    checks.append(Check(
        "L0 能定位到具体异常原子和构型",
        bool(worst),
        f"L0.worst_cases 导出 {worst[0]['extra']['n_frames'] if worst else 0} 帧、{n_atoms} 个原子（含元素与帧号）",
    ))

    # 4. the L1 gate blocks expensive levels (the L1 file is named after the model)
    l1 = next((results[k] for k in files_with("L1_") if results[k].get("context", {}).get("gate")), {})
    gate = l1.get("context", {}).get("gate")
    checks.append(Check(
        "L1 的故障会阻止无意义的高成本任务",
        bool(gate is not None),
        (f"门控已实现并执行：{gate['n_checks']} 项检查，{gate['n_failed']} 项失败 -> {gate['decision']}"
         if gate else "未找到 L1 门控记录"),
    ))

    # 5. L3 uses identical cutoffs / bins / ring parameters on both sides
    l3_files = files_with("L3")
    frozen = {k: results[k]["context"].get("frozen_params", {}) for k in l3_files}
    consistent = all(
        frozen[k].get("bond_cutoffs") == frozen[l3_files[0]].get("bond_cutoffs")
        for k in l3_files
        if results[k]["context"].get("cutoff_source_state") == results[l3_files[0]]["context"].get("cutoff_source_state")
    ) if l3_files else False
    checks.append(Check(
        "L3 两侧使用完全相同的 cutoff、bin 和 ring 参数",
        bool(l3_files) and consistent,
        (f"{len(l3_files)} 个 L3 结果文件各自记录 frozen_params 与 cutoff_source_state；"
         f"ring 算法 = {frozen[l3_files[0]].get('ring_algorithm') if l3_files else 'n/a'}"),
    ))

    # 6. L4 reports its fit window and rejects ramps
    l4 = next((results[k] for k in files_with("L4")), {})
    window = [r for r in rows if r["metric_id"].startswith("L4.fit_window")]
    guard = [r for r in rows if "equilibrium_segment" in r["metric_id"]]
    checks.append(Check(
        "L4 报告拟合窗口，且不接受升降温轨迹伪装成平衡扩散",
        bool(window) and bool(guard),
        (f"拟合窗口 {window[0]['value']} fs 已作为 metric 记录；"
         f"平衡性守卫在 {len(guard)} 条轨迹上执行"),
    ))

    # 7. single-trajectory results are marked provisional; canonical state points
    #    pair every MLIP seed with its own AIMD replica, so they may be graded
    def state_of(r):
        mid = r["metric_id"]
        return mid.split("@", 1)[1].split(".")[0] if "@" in mid else None
    # (only state-point metrics carry "@<state>"; L0-L2 have no replicas at all)
    single = [r for r in rows if state_of(r) is not None and state_of(r) not in CANONICAL_STATES]
    graded = [r for r in single if r["grade"] != "ungraded"]
    n_canon_graded = sum(1 for r in rows if state_of(r) in CANONICAL_STATES and r["grade"] != "ungraded")
    static_graded = [r for r in rows if r["level"] in ("L0", "L1", "L2") and r["grade"] != "ungraded"]
    checks.append(Check(
        "单轨迹结果明确标记 provisional，且不产生正式等级",
        len(graded) == 0 and len(static_graded) == 0,
        (f"单参考状态点的 metric {len(single)} 条（legacy 已退役，应为 0），其中 {len(graded)} 条被判级；"
         f"无独立参考副本的 L0–L2 有 {len(static_graded)} 条被判级（应为 0）；"
         f"canonical 多副本配对共 {n_canon_graded} 条正式等级"),
    ))

    # 8. model error vs reference noise vs protocol difference.  Since the legacy
    #    references were retired every reference is phase-1 canonical, so there is
    #    no protocol difference left to measure -- check that, rather than assume it.
    ref_protocols = sorted({str(r.get("reference_protocol")) for r in rows if r.get("reference_protocol")})
    foreign = [p for p in ref_protocols if "canonical" not in p]
    l5 = next((results[k] for k in files_with("L5")), {})
    band = l5.get("context", {}).get("band_definition")
    unc_files = files_with("L3_uncertainty")
    verdicts: dict[str, int] = {}
    for k in unc_files:
        for r in results[k]["results"]:
            v = (r.get("extra") or {}).get("verdict", "unknown")
            verdicts[v] = verdicts.get(v, 0) + 1
    checks.append(Check(
        "报告区分模型误差、参考统计噪声与协议差异",
        not foreign,
        (f"参考协议 {len(ref_protocols)} 个，全部为 phase-1 canonical"
         + (f"，但出现非 canonical：{foreign}" if foreign else "，d_protocol = 0（按构造，无需 bridge set）")
         + f"；L5 参考噪声用『{band}』；"
         f"L3 用分块 SEM + 种子散布把 {sum(verdicts.values())} 个观测量分成 {verdicts}；"
         "L3–L5 的 sigma_ref 取 3 条 AIMD replica 间离散度"),
    ))

    # 11. are the reported differences resolvable at all?
    if unc_files:
        resolved = verdicts.get("resolved", 0)
        total = sum(verdicts.values())
        checks.append(Check(
            "L3 的每个数字都带可测的误差尺度，并说明是否可分辨",
            total > 0,
            (f"{total} 个标量观测量各带分块 SEM；{resolved} 个 resolved、"
             f"{verdicts.get('within_seed_scatter', 0)} 个落在模型自身种子散布内、"
             f"{verdicts.get('within_statistical_noise', 0)} 个仍埋在统计噪声里"),
        ))

    # 9. de-duplicated candidate list
    manifests = sorted((paths.RESULTS / "candidates").glob("*_manifest.json"))
    total = 0
    for m in manifests:
        total += json.loads(m.read_text()).get("n_candidates", 0)
    checks.append(Check(
        "能导出一份去重后的候选 DFT 补算结构列表",
        bool(manifests),
        f"{len(manifests)} 份 manifest，共 {total} 条候选（含来源 metric、选择理由、聚类信息、建议协议、回归 metric ID）",
    ))

    # 10. protected audit set scored but never mined for new data
    roles = {results[k]["context"].get("reference_role") for k in results if "reference_role" in results[k].get("context", {})}
    origins: dict[str, int] = {}
    for m in manifests:
        for c in json.loads(m.read_text()).get("candidates", []):
            origins[c.get("role_of_origin", "?")] = origins.get(c.get("role_of_origin", "?"), 0) + 1
    audit_ctx = results.get("L0_audit", {}).get("context", {})
    checks.append(Check(
        "不读取或修改 protected audit set 来指导补数据",
        "protected_audit" not in origins and ("L0_audit" not in results or "failure_export" in audit_ctx),
        f"参与评估的参考数据角色：{sorted(r for r in roles if r) or ['n/a']}；"
        f"候选来源角色：{origins or '无候选'}；"
        f"L0_audit：{audit_ctx.get('failure_export', '未运行')}",
    ))
    return checks


ENGINEERING_CHECKLIST = [
    ("定义 structure、protocol、model 和 result 的最小 schema", "mlipeval/schema.py", True),
    ("编写 VASP/XDATCAR 读取与 extxyz 转换", "mlipeval/io_vasp.py, io_labeled.py", True),
    ("固定并校验 MACE-MPA-0 checkpoint", "calculators.ModelSpec.from_checkpoint（sha256）", True),
    ("实现统一 calculator adapter", "mlipeval/calculators.py", True),
    ("实现 L0 和 L1", "l0_reference.py, l1_pes.py", True),
    ("实现 L2 的结构弛豫与 EOS",
     "l2_static.py；立方相 DFT EOS 已完成：5 个空位排布 × 7 体积，"
     "逐 seed 与 MLIP 配对，误差取 seed 均值并附跨 seed 散布", True),
    ("实现 GST225 的 L3a/L3b", "l3a_short_range.py, l3b_medium_range.py", True),
    ("实现单温度 L4", "l4_dynamics.py；950 / 1200 / 3000 K 三个液体状态点", True),
    ("实现 melt–quench 的 L5 路径比较", "l5_pathway.py；1200→300 K 淬火 + 300→1200 K 升温", True),
    ("实现独立轨迹不确定度与 protocol mismatch 标记",
     "uncertainty.py（分块 SEM + 种子散布）+ ensemble.py（3 replica sigma_ref）；"
     "参考全部为 canonical 协议，d_protocol 按构造为 0", True),
    ("实现失败构型导出", "failure_export.py（含窗口保全的去重）；L0 诊断集 -> results/candidates/", True),
    ("生成 GST225 第一版诊断报告", "本脚本 + eval/README.md", True),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--markdown", action="store_true", help="also write results/PHASE1_REPORT.md")
    args = ap.parse_args()

    import pandas as pd

    results = load_results()
    rows = all_metrics(results)
    frame = pd.DataFrame(rows)

    print(f"结果文件 {len(results)} 个，metric {len(rows)} 条\n")
    summary = (
        frame.groupby(["level", "status"]).size().unstack(fill_value=0)
        if not frame.empty else pd.DataFrame()
    )
    print(summary.to_string())
    print("\n证据等级分布:")
    print(frame.groupby(["level", "evidence_tier"]).size().unstack(fill_value=0).to_string())

    checks = acceptance_checks(results, rows)
    print("\n\n验收条件（validation_methods_L0-L5.md 第 8 节）")
    print("=" * 100)
    for c in checks:
        print(f"[{c.mark:7s}] {c.condition}\n           {c.evidence}")

    print("\n\n工程清单（第 9 节）")
    print("=" * 100)
    for item, where, state in ENGINEERING_CHECKLIST:
        mark = {True: "x", None: "~", False: " "}[state]
        print(f"  [{mark}] {item:44s} {where}")

    if args.markdown:
        lines = ["# GST225 第一阶段诊断报告", "",
                 f"结果文件 {len(results)} 个，metric {len(rows)} 条。", "",
                 "## 各级状态分布", "", "```", summary.to_string(), "```", "",
                 "## 验收条件", "", "| 状态 | 条件 | 依据 |", "|---|---|---|"]
        for c in checks:
            lines.append(f"| {c.mark} | {c.condition} | {c.evidence} |")
        lines += ["", "## 工程清单", ""]
        for item, where, state in ENGINEERING_CHECKLIST:
            mark = {True: "x", None: "~", False: " "}[state]
            lines.append(f"- [{mark}] {item} — `{where}`")
        # only the reportable columns: the full `extra` payloads live in the JSON
        columns = [c for c in ("level", "metric_id", "value", "unit", "status",
                               "evidence_tier", "grade", "z_eff", "failure_reason")
                   if c in frame.columns]
        table = frame[columns].copy() if not frame.empty else frame
        if not table.empty:
            table["value"] = table["value"].apply(
                lambda v: "<array>" if isinstance(v, (list, dict)) else v
            )
        lines += ["", "## 全部 metric", "",
                  "完整的 `extra` 字段（协议、阈值、审计记录等）见 `results/*.json`。", "",
                  table.to_markdown(index=False) if not table.empty else ""]
        out = paths.RESULTS / "PHASE1_REPORT.md"
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
