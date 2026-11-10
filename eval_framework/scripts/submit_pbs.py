#!/usr/bin/env python
"""Submit the MACE-heavy parts of the evaluation as PBS jobs.

    python submit_pbs.py --chain                 # 整条依赖链，一次提交
    python submit_pbs.py --chain --l345 --all-models   # 只重跑 MD + L3-L5（状态点切换后）
    python submit_pbs.py --command "run_l0.py"   # 单个 driver
    python submit_pbs.py --chain --dry-run       # 只生成 .pbs，不提交

Project rule (``CLAUDE.md``): MACE and MD work never runs on the login node.
The heavy drivers enforce that themselves via
:func:`mlipeval.pbs.require_compute_node`; this script is the sanctioned way to
get them onto a GPU compute node.

Job scripts are written to ``eval/scripts/pbs/`` and their output to
``eval/results/pbs/``, so a failed job can be inspected without rerunning it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mlipeval import models
from mlipeval.pbs import DEFAULT_SELECT, describe_environment, submit, write_job
from mlipeval.state_points import analysis_commands, md_commands

#: The dependency chain, in the order the framework requires.
#: L1 is a gate, so it is submitted before anything expensive; the analysis-only
#: drivers are folded into the job that produces the data they consume, which
#: keeps the number of queued jobs small.
# Walltimes are sized from the H200 probe (probe_gpu.py): 263 ms per 216-atom
# single point, 36 step/s for 306-atom MD -- roughly 36x the login node.  They
# are then given a wide margin, because a queued job that dies on walltime costs
# far more than an over-request.
#: Walltime multiplier per model, from the LAMMPS smoke test throughput
#: (BUILD_AND_RUN.md): MACE 10.1, NequIP 13.5, MatterSim 11.7, CHGNet 6.0 and
#: DPA-4 0.79 katom-step/s on one H200.  A queued job that dies on walltime
#: costs far more than an over-request, so these round generously upward.
SLOWDOWN = {
    "mace_mpa0": 1.0,
    # same architecture as MPA-0 by construction (the specialised potentials are
    # aligned to it item by item), so the same cost per evaluation
    "mace_spec_gst": 1.0,
    "nequip_oam_l": 1.5,
    # architecture read off OAM-L item by item, so the same cost per evaluation
    "nequip_spec_gst": 1.5,
    "mattersim_5m": 1.5,
    "chgnet_030": 2.5,
    # DPA-4 is 13x slower than MACE through `pair_style deepmd` in LAMMPS but
    # *faster* than it through ASE (306-atom single point: 0.19 s vs 0.31 s), so
    # that LAMMPS figure is pair-style overhead, not the model.  Sized from the
    # ASE probe instead.
    "dpa4_pro": 2.0,
}


def _scale(walltime: str, factor: float) -> str:
    h, m, sec = (int(x) for x in walltime.split(":"))
    total = int((h * 3600 + m * 60 + sec) * factor)
    total = min(total, 96 * 3600)
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def chain_for(model_key: str | None, *, l345_only: bool = False) -> list[tuple[str, list[str], str]]:
    """The dependency chain, in the order the framework requires.

    L1 is a gate, so it is submitted before anything expensive; the analysis-only
    drivers are folded into the job that produces the data they consume, which
    keeps the number of queued jobs small.

    Walltimes are sized from the H200 probe (probe_gpu.py): 263 ms per 216-atom
    single point, 36 step/s for 306-atom MD, then scaled by :data:`SLOWDOWN`.

    ``warm_cache.py`` only feeds the phase-1 notebooks, which read the flat MACE
    layout.

    The MD stage runs the six canonical state points with three seeds, one per
    AIMD replica: 3 x (10000 + 10000 + 8333 + 20000 + 3000 + 8333) = 179k steps
    of 306-atom MD, about 95 min for MACE-MPA-0 at the measured 32 step/s.
    ``l345_only`` drops L0-L2, which do not depend on the state points.
    """
    analysis = analysis_commands() + ["run_report.py --markdown"]
    if model_key is None:  # flat phase-1 layout also refills the notebook caches
        analysis.append("warm_cache.py")

    chain = [
        ("l0", ["run_l0.py"], "02:00:00"),
        ("l1", ["run_l1.py"], "03:00:00"),
        ("l2", ["run_l2.py"], "04:00:00"),
        ("md", md_commands(), "12:00:00"),
        ("analysis", analysis, "06:00:00"),
    ]
    if l345_only:
        chain = chain[3:]
    factor = SLOWDOWN.get(model_key or "mace_mpa0", 2.0)
    prefix = f"mev_{model_key}" if model_key else "mlipeval"
    return [
        (f"{prefix}_{stage}", cmds, _scale(wt, factor if stage != "analysis" else 1.5))
        for stage, cmds, wt in chain
    ]


def _sum_walltime(walltimes: list[str], *, margin: float = 1.0) -> str:
    """Total walltime for the merged job, capped at the queue's 144 h limit."""
    total = 0
    for wt in walltimes:
        h, m, s = (int(x) for x in wt.split(":"))
        total += h * 3600 + m * 60 + s
    total = min(int(total * margin), 144 * 3600)
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def merged_chain(model_key: str | None, *, l345_only: bool = False) -> tuple[str, list[str], str]:
    """The whole chain as one job: same commands, same order, one queue wait.

    Five afterok-linked jobs mean five separate waits for a GPU, and on this
    cluster the wait dominates: the L0-L5 drivers for one model add up to a few
    hours of compute against a queue that can sit for longer than that per job.
    The dependency edges buy nothing here -- the stages are strictly sequential
    on one node either way -- so merging trades no parallelism for four fewer
    waits.

    The ordering guarantee that ``afterok`` provided is preserved inside the job
    by :func:`mlipeval.pbs._render_commands`, which stops at the first failing
    step and exits with its code.

    Kept as a mode rather than the default because the split chain is still the
    better shape when a stage is being re-run on its own, and because a merged
    job that dies on walltime loses every stage after the one that overran.
    """
    stages = chain_for(model_key, l345_only=l345_only)
    commands = [c for _, cmds, _ in stages for c in cmds]
    walltime = _sum_walltime([wt for _, _, wt in stages])
    prefix = f"mev_{model_key}" if model_key else "mlipeval"
    return f"{prefix}_{'l345' if l345_only else 'all'}", commands, walltime


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--chain", action="store_true", help="submit the whole dependency chain")
    group.add_argument("--command", help="submit one driver invocation, e.g. \"run_l0.py --refresh\"")
    ap.add_argument("--name", default=None, help="job name for --command")
    ap.add_argument("--walltime", default="06:00:00")
    ap.add_argument("--select", default=DEFAULT_SELECT)
    ap.add_argument("--queue", default=None,
                    help="PBS queue; left unset by default (this cluster routes through 'auto')")
    ap.add_argument("--dry-run", action="store_true", help="write the .pbs files but do not qsub")
    ap.add_argument("--no-depend", action="store_true",
                    help="submit chain jobs independently instead of afterok-chained")
    # any registered model, not just ORDER: the specialised potentials are
    # deliberately outside the universal ranking but still run the same drivers
    ap.add_argument("--model", default=None, choices=models.ORDER + models.SPECIALISED,
                    help="model under test; sets MLIPEVAL_MODEL and the conda env "
                         "(default: the flat phase-1 MACE layout)")
    ap.add_argument("--all-models", action="store_true",
                    help="submit one independent chain per universal model (ORDER)")
    ap.add_argument("--l345", action="store_true",
                    help="with --chain: only the MD and L3-L5 analysis stages (L0-L2 do not "
                         "depend on the state points)")
    ap.add_argument("--split-jobs", action="store_true",
                    help="submit the chain as five afterok-linked jobs instead of one "
                         "merged job; five queue waits rather than one, so use it only "
                         "when re-running a single stage matters more than turnaround")
    args = ap.parse_args()

    env = describe_environment()
    print(f"submitting from {env['hostname']} (in_pbs_job={env['in_pbs_job']})")

    if args.command:
        prefix = f"mev_{args.model}_" if args.model else "mlipeval_"
        name = args.name or (prefix + args.command.split()[0].removeprefix("run_").removesuffix(".py"))
        job = write_job(name, args.command, walltime=args.walltime, select=args.select,
                        queue=args.queue, model=args.model)
        job_id = submit(job, dry_run=args.dry_run)
        print(f"  {job.path.name:28s} -> {job_id or '(dry run)'}   log: {job.logfile}")
        return 0

    keys = models.ORDER if args.all_models else [args.model]
    for key in keys:
        if len(keys) > 1:
            print(f"\n--- {key} ---")
        if not args.split_jobs:
            name, commands, walltime = merged_chain(key, l345_only=args.l345)
            job = write_job(name, commands, walltime=walltime, select=args.select,
                            queue=args.queue, model=key)
            job_id = submit(job, dry_run=args.dry_run)
            print(f"  {job.path.name:28s} -> {job_id or '(dry run)'}")
            print(f"      commands: {len(commands)} (整条链合并)   walltime {walltime}"
                  f"   log: {job.logfile}")
            print(f"      stages: {', '.join(s for s, _, _ in chain_for(key, l345_only=args.l345))}")
            continue

        previous: str | None = None
        chain = chain_for(key, l345_only=args.l345)
        for name, commands, walltime in chain:
            job = write_job(name, commands, walltime=walltime, select=args.select,
                            queue=args.queue, model=key)
            depends = None if (args.no_depend or args.dry_run) else previous
            job_id = submit(job, depends_on=depends, dry_run=args.dry_run)
            arrow = f" (after {depends})" if depends else ""
            print(f"  {job.path.name:28s} -> {job_id or '(dry run)'}{arrow}")
            print(f"      commands: {len(commands)}   walltime {walltime}   log: {job.logfile}")
            previous = job_id or previous

    print(f"\njob scripts: {Path(__file__).resolve().parent / 'pbs'}")
    print("查看状态: qstat -u $USER      取消: qdel <jobid>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
