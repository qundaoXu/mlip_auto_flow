#!/usr/bin/env python
"""Run the whole phase-1 diagnostic chain in dependency order.

    python run_all.py [--threads T] [--skip-md] [--quick]

This is meant to run **inside a PBS job**, not on the login node -- it drives
every MACE-heavy stage.  Use ``submit_pbs.py --chain`` to get the same sequence
onto GPU nodes as separate, dependency-chained jobs (which is preferable: a
failure then costs one stage rather than the whole run).

Order matters: L1 is a gate.  If any L1 check fails, L4 and L5 are skipped
rather than run, which is the behaviour ``validation_methods_L0-L5.md`` asks
for -- expensive levels must not run on a model with a pathological PES.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from mlipeval import paths
from mlipeval.pbs import in_pbs_job
from mlipeval.state_points import analysis_commands


def run(script: str, *args: str) -> int:
    cmd = [sys.executable, "-u", str(HERE / script), *args]
    print(f"\n{'=' * 78}\n$ {' '.join(cmd[1:])}\n{'=' * 78}", flush=True)
    return subprocess.call(cmd)


def l1_passed() -> bool:
    """Whether *this* model's L1 gate is open.

    The filename has to follow ``MLIPEVAL_MODEL``: results are written as
    ``L1_<key>.json`` under ``results/models/<key>/``, so hard-coding
    ``L1_mace_mpa0.json`` found nothing for any of the other five models and
    reported their gate as closed no matter what L1 actually said.  It failed
    closed, so it never produced a wrong number -- it would just have stopped
    every non-MACE run before L2.
    """
    from mlipeval import models

    path = paths.RESULTS / f"L1_{models.current_key()}.json"
    if not path.exists():
        print(f"L1 gate: no result file at {path}", file=sys.stderr)
        return False
    return bool(json.loads(path.read_text())["context"]["gate"]["pass"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--skip-md", action="store_true", help="reuse existing eval/data/md_*.npz")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--allow-login-node", action="store_true",
                    help="bypass the compute-node check; only for trivial smoke tests")
    ap.add_argument("--seeds", type=int, default=3,
                    help="MLIP MD seeds per state point; the canonical state points pair seed k "
                         "with AIMD replica k+1, so all three are needed for an ensemble grade")
    args = ap.parse_args()
    if not in_pbs_job() and not args.allow_login_node:
        print(
            "run_all.py 会驱动全部 MACE 计算，不能在登录节点运行。\n"
            "请改用：  python submit_pbs.py --chain\n"
            "（该命令把同一条链拆成带 afterok 依赖的 GPU 作业，单个阶段失败也不会拖垮整条链）",
            file=sys.stderr,
        )
        return 3
    threads = ["--threads", str(args.threads)] if args.threads else []

    if run("run_l0.py", *threads):
        return 1
    if run("run_l1.py", *threads, *(["--quick"] if args.quick else [])):
        return 1

    if not l1_passed():
        print("\nL1 gate is CLOSED -- stopping before L2-L5, as the framework requires.")
        return 2
    print("\nL1 gate is OPEN -- continuing to L2-L5.")

    if run("run_l2.py", *threads):
        return 1
    if not args.skip_md:
        for seed in range(max(1, args.seeds)):
            extra = ["--seed", str(seed)] if seed else []
            if run("run_mlip_md.py", "--state", "all", *extra, *threads):
                return 1
    # L3-L5 come from the one shared list (state_points.analysis_commands), so
    # this driver and the PBS chain cannot drift apart.
    for script, extra in (
        *((cmd.split()[0], cmd.split()[1:]) for cmd in analysis_commands()),
        ("run_report.py", ["--markdown"]),
    ):
        if run(script, *extra):
            return 1

    print(f"\nAll levels complete. Results in {paths.RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
