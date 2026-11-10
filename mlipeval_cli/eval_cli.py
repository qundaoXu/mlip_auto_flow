"""``mlipeval`` — command-line entry point of the L0–L5 evaluation framework.

Every subcommand either forwards to an existing driver in
``eval_framework/scripts/`` (so the CLI and the notebooks/PBS chains run the
same code) or reads result files that those drivers wrote.  Unknown options are
passed through to the driver unchanged, e.g.::

    mlipeval run l3 --model mace_mpa0 --state canon_amorphous_300K
    mlipeval run l0 --model nequip_oam_l --set audit --refresh

Exit codes: the driver's own exit code is returned; 2 means bad usage.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import _paths

LEVEL_DRIVERS = {
    "l0": "run_l0.py",
    "l1": "run_l1.py",
    "l2": "run_l2.py",
    "md": "run_mlip_md.py",
    "l3": "run_l3.py",
    "l3u": "run_l3_uncertainty.py",
    "l4": "run_l4.py",
    "l5": "run_l5.py",
    "all": "run_all.py",
}
LEAKAGE_DRIVERS = {
    "h5": "screen_h5_leakage.py",
    "diagnostic": "screen_diagnostic_leakage.py",
    "al-ladder": "screen_al_ladder.py",
}


def _driver(script: str, args: list[str], model: str | None = None) -> int:
    scripts = _paths.eval_scripts()
    path = scripts / script
    if not path.exists():
        sys.exit(f"mlipeval: driver not found: {path}")
    env = dict(os.environ)
    if model:
        env["MLIPEVAL_MODEL"] = model
    cmd = [sys.executable, str(path), *args]
    print("+", ("MLIPEVAL_MODEL=" + model + " ") if model else "", " ".join(cmd[1:]), file=sys.stderr)
    return subprocess.call(cmd, cwd=str(scripts), env=env)


# --------------------------------------------------------------------------- models
def cmd_models(ns, extra) -> int:
    pkg = _paths.import_mlipeval()
    from mlipeval import models, paths  # noqa: F401

    groups = {k: "universal" for k in models.ORDER}
    for k in models.SPECIALISED:
        groups[k] = "al_ladder" if k.startswith("al_") else "specialised"
    rows = []
    for key in list(models.ORDER) + [k for k in models.SPECIALISED if k not in models.ORDER]:
        e = models.get(key)
        rows.append({
            "key": e.key,
            "label": e.label,
            "group": groups.get(e.key, "other"),
            "backend": e.backend,
            "dtype": e.dtype,
            "protocol": e.native_protocol,
            "has_results": (paths.MODEL_RESULTS_ROOT / e.key).is_dir(),
        })
    if ns.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    w = max(len(r["key"]) for r in rows)
    for r in rows:
        flag = "✓" if r["has_results"] else " "
        print(f"{flag} {r['key']:<{w}}  {r['group']:<11}  {r['backend']:<9} {r['label']}")
    return 0


# --------------------------------------------------------------------------- run / submit
def cmd_run(ns, extra) -> int:
    return _driver(LEVEL_DRIVERS[ns.level], extra, model=ns.model)


def cmd_submit(ns, extra) -> int:
    if ns.al_ladder:
        return _driver("submit_al_ladder.py", extra)
    args = list(extra)
    if ns.model:
        args = ["--model", ns.model, *args]
    return _driver("submit_pbs.py", args)


# --------------------------------------------------------------------------- rank / card
def _build_ranking(keys: list[str], clean: bool, weights: str | None):
    _paths.import_mlipeval()
    from mlipeval import paths
    from mlipeval.ranking import (INDICATORS, LEVEL_WEIGHTS, build_ranking,
                                  contaminated_metric_ids, gate_is_contaminated, load_results)

    keys = [k for k in keys if (paths.MODEL_RESULTS_ROOT / k).is_dir()]
    if not keys:
        sys.exit("mlipeval: none of the requested models has results under results/models/")
    lw = None
    if weights:
        lw = dict(LEVEL_WEIGHTS)
        for item in weights.split(","):
            k, _, v = item.partition("=")
            lw[k.strip().upper()] = float(v)
    exclude: frozenset[str] = frozenset()
    indicators = INDICATORS
    if clean:
        exclude = frozenset(contaminated_metric_ids(keys))
        if any(gate_is_contaminated(load_results(k)) for k in keys):
            indicators = tuple(i for i in INDICATORS if i.key != "L1 gate")
    return build_ranking(keys, level_weights=lw, exclude_metrics=exclude, indicators=indicators), keys, exclude


def _default_keys() -> list[str]:
    _paths.import_mlipeval()
    from mlipeval import models
    return list(models.ORDER) + [k for k in models.SPECIALISED if not k.startswith("al_")]


def cmd_rank(ns, extra) -> int:
    args = list(extra)
    if ns.models:
        args += ["--models", ns.models]
    if ns.clean:
        args.append("--drop-contaminated")
    if ns.markdown:
        args.append("--markdown")
    if ns.out_stem:
        args += ["--out-stem", ns.out_stem]
    if ns.level_weights:
        args += ["--level-weights", ns.level_weights]
    if ns.models and not ns.out_stem:
        print("mlipeval: note — a --models subset without --out-stem overwrites results/ranking.json",
              file=sys.stderr)
    return _driver("run_ranking.py", args)


def cmd_card(ns, extra) -> int:
    """Machine-readable report card for one model (read-only, no GPU)."""
    keys = ns.against.split(",") if ns.against else _default_keys()
    if ns.model not in keys:
        keys.append(ns.model)
    ranking, keys, exclude = _build_ranking(keys, ns.clean, ns.level_weights)
    from mlipeval import models, paths
    from mlipeval.ranking import LEVELS, load_results

    order = ranking.order()
    res = load_results(ns.model)
    l1 = next((v for k, v in res.items() if k.startswith("L1")), None) or {}
    gate = (l1.get("context") or {}).get("gate", {})
    cands = []
    for mf in sorted((paths.MODEL_RESULTS_ROOT / ns.model / "candidates").glob("*manifest.json")):
        cands += json.loads(mf.read_text()).get("candidates", [])
    card = {
        "model": ns.model,
        "label": models.get(ns.model).label,
        "compared_against": keys,
        "scoring": "clean" if ns.clean else "all",
        "excluded_metrics": len(exclude),
        "overall": ranking.overall[ns.model],
        "rank": order.index(ns.model) + 1,
        "n_models": len(order),
        "level_scores": {lv: ranking.level_scores[lv][ns.model] for lv in LEVELS},
        "l1_gate": {"pass": bool(gate.get("pass", False)), "failed": gate.get("failed_metrics", []),
                     "decision": gate.get("decision")},
        "indicators": [
            {"level": r.indicator.level, "key": r.indicator.key, "unit": r.indicator.unit,
             "raw": r.raw.get(ns.model), "score": r.score.get(ns.model)}
            for r in ranking.rows
        ],
        "candidates": [{"id": c["candidate_id"], "metric": c["source_metric"], "reason": c["reason"],
                        "protocol": c["suggested_protocol"]} for c in cands],
        "result_files": sorted(str(p) for p in (paths.MODEL_RESULTS_ROOT / ns.model).glob("*.json")),
    }
    if ns.json:
        print(json.dumps(card, ensure_ascii=False, indent=2, default=float))
        return 0
    print(f"{card['label']}  —  overall {card['overall']:.1f}  (rank {card['rank']}/{card['n_models']}, "
          f"{card['scoring']} scoring)")
    for lv, v in card["level_scores"].items():
        print(f"  {lv}  {'—' if v is None else f'{v:5.1f}'}")
    g = card["l1_gate"]
    print(f"  L1 gate: {'pass' if g['pass'] else 'closed'}"
          + ("" if g["pass"] else f" ({', '.join(g['failed'])})"))
    print(f"  candidates for the next labelling round: {len(card['candidates'])}")
    return 0


# --------------------------------------------------------------------------- candidates / leakage / report
def cmd_candidates(ns, extra) -> int:
    _paths.import_mlipeval()
    from mlipeval import paths

    src = paths.MODEL_RESULTS_ROOT / ns.model / "candidates"
    manifests = sorted(src.glob("*manifest.json"))
    if not manifests:
        print(f"no candidates exported for {ns.model} (run `mlipeval run l0 --model {ns.model}` first)")
        return 1
    for mf in manifests:
        m = json.loads(mf.read_text())
        print(f"{mf.name}: {m['n_candidates']} candidates (from {m['n_before_dedup']}, {m['dedup_method']})")
        for c in m["candidates"]:
            print(f"  {c['candidate_id']:<14} {c['source_metric']:<16} {c['reason']}")
    if ns.out:
        out = Path(ns.out)
        out.mkdir(parents=True, exist_ok=True)
        for p in src.iterdir():
            shutil.copy2(p, out / p.name)
        print(f"copied {src} -> {out}")
    return 0


def cmd_leakage(ns, extra) -> int:
    return _driver(LEAKAGE_DRIVERS[ns.target], extra)


def cmd_report(ns, extra) -> int:
    return _driver("run_report.py", extra + (["--markdown"] if ns.markdown else []), model=ns.model)


# --------------------------------------------------------------------------- config
def cmd_config(ns, extra) -> int:
    """Resolved site configuration; ``--check`` also verifies the paths."""
    if ns.init:
        src = _paths.RELEASE_ROOT / "config" / "mlip_auto_flow.example.yaml"
        dst = Path(ns.init)
        if dst.exists():
            sys.exit(f"mlipeval config --init: {dst} already exists; not overwriting")
        shutil.copy2(src, dst)
        print(f"wrote {dst} -- edit it, then run `mlipeval config --check`")
        return 0
    _paths.import_mlipeval()
    from mlipeval import config, models

    summ = config.summary()
    checks = []
    if ns.check:
        for k, v in summ.items():
            if k.startswith(("paths.", "lammps.", "environment.")) and isinstance(v, str) and v.startswith("/"):
                checks.append({"item": k, "path": v, "exists": Path(v).exists()})
        for key in list(models.ORDER) + [k for k in models.SPECIALISED if k not in models.ORDER]:
            e = models.get(key)
            for field in ("checkpoint", "conda_env", "lammps_pair_in"):
                v = getattr(e, field)
                if v is not None:
                    checks.append({"item": f"models.{key}.{field}", "path": str(v), "exists": Path(v).exists()})
    if ns.json:
        print(json.dumps({"settings": summ, "checks": checks}, ensure_ascii=False, indent=2, default=str))
    else:
        w = max(len(k) for k in summ)
        for k, v in summ.items():
            print(f"{k:<{w}}  {v}")
        if checks:
            missing = [c for c in checks if not c["exists"]]
            print(f"\n{len(checks) - len(missing)}/{len(checks)} paths exist")
            for c in missing:
                print(f"  missing  {c['item']:<44} {c['path']}")
    return 1 if ns.check and any(not c["exists"] for c in checks) and ns.strict else 0


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mlipeval", description="L0–L5 evaluation of machine-learning interatomic potentials")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("config", help="show the resolved site configuration (mlip_auto_flow.yaml + env)")
    s.add_argument("--check", action="store_true", help="also check that every configured path exists")
    s.add_argument("--strict", action="store_true", help="with --check: exit 1 if anything is missing")
    s.add_argument("--json", action="store_true")
    s.add_argument("--init", metavar="PATH", default=None, help="write a config template to PATH")
    s.set_defaults(fn=cmd_config)

    s = sub.add_parser("models", help="list registered models and whether they have results")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_models)

    s = sub.add_parser("run", help="run one evaluation level for one model (forwards to run_<level>.py)")
    s.add_argument("level", choices=list(LEVEL_DRIVERS))
    s.add_argument("--model", required=True, help="model key from `mlipeval models`")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("submit", help="submit evaluation jobs to PBS (forwards to submit_pbs.py)")
    s.add_argument("--model", default=None)
    s.add_argument("--al-ladder", action="store_true", help="L0–L2 for every AL-ladder model (submit_al_ladder.py)")
    s.set_defaults(fn=cmd_submit)

    s = sub.add_parser("rank", help="cross-model relative ranking (forwards to run_ranking.py)")
    s.add_argument("--models", default=None, help="comma-separated keys")
    s.add_argument("--clean", action="store_true", help="drop metrics any listed model was trained on")
    s.add_argument("--markdown", action="store_true")
    s.add_argument("--out-stem", default=None)
    s.add_argument("--level-weights", default=None, help='e.g. "L3=0.35,L5=0.05"')
    s.set_defaults(fn=cmd_rank)

    s = sub.add_parser("card", help="report card for one model, read-only; --json for platforms")
    s.add_argument("--model", required=True)
    s.add_argument("--against", default=None, help="comma-separated keys to rank against "
                   "(default: universal + specialised models)")
    s.add_argument("--clean", action="store_true")
    s.add_argument("--level-weights", default=None)
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_card)

    s = sub.add_parser("candidates", help="show / export failure-driven candidate structures")
    s.add_argument("--model", required=True)
    s.add_argument("--out", default=None, help="copy extxyz + manifest here")
    s.set_defaults(fn=cmd_candidates)

    s = sub.add_parser("leakage", help="screen training sets against the reference data")
    s.add_argument("target", choices=list(LEAKAGE_DRIVERS))
    s.set_defaults(fn=cmd_leakage)

    s = sub.add_parser("report", help="phase-1 acceptance report for one model (run_report.py)")
    s.add_argument("--model", required=True)
    s.add_argument("--markdown", action="store_true")
    s.set_defaults(fn=cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ns, extra = parser.parse_known_args(argv)
    if extra and ns.cmd in ("config", "models", "card", "candidates"):
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    if extra and extra[0] == "--":
        extra = extra[1:]
    return int(ns.fn(ns, extra) or 0)


if __name__ == "__main__":
    sys.exit(main())
