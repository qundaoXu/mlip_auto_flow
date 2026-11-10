#!/usr/bin/env python
"""Submit L0-L2 for every point of the AL ladder, one job per model.

    python submit_al_ladder.py [--dry-run] [--levels l0,l1,l2] [--models ...]

Why not ``submit_pbs.py --chain --model al_gen0`` eight times: the merged chain
includes ``run_mlip_md.py``, which needs a LAMMPS ``pair.in`` and an mliap export
per model.  L0-L2 need neither -- they go through the ASE calculator on the
native ``.model`` -- and they are the levels that answer the question the ladder
was built to ask, so they run first and on their own.

One job per model rather than one per level: GPUs on this cluster are close to
fully booked, so three queue waits per model would dominate the ~4 minutes of
GPU time the three levels actually need.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from mlipeval import models, paths
from mlipeval.pbs import submit, write_job
from mlipeval.state_points import analysis_commands, md_commands

LEVELS = {"l0": "run_l0.py", "l1": "run_l1.py", "l2": "run_l2.py"}

#: Everything downstream of L2, as one job: the mliap export, both MD seeds, and
#: the analysis drivers that read the trajectories back. ``submit_pbs.chain_for``
#: is not reused because it also carries l0-l2 (already done) and because the
#: export step is specific to this ladder.
EXPORT = str(paths.AL_LADDER_EXPORT)  # lammps.al_ladder_export in the site config
MD_STAGE = [*md_commands(), *analysis_commands(), "run_report.py --markdown"]
# measured per model on an H200 (README): L0 0:39, L1 2:00, L2 0:56.
# The ladder models are 0.75M-parameter MACE, i.e. smaller than MPA-0, so these
# are upper bounds; the margin is deliberate because a walltime kill costs a
# whole queue wait.
WALLTIME = "02:00:00"
#: Three seeds of the canonical state points: ~145k MD steps, about 75 min for
#: MACE-MPA-0, plus the replica-ensemble analysis.  The ladder models are 0.75M
#: parameters against MPA-0's larger network, so this is generous.
MD_WALLTIME = "12:00:00"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(models.AL_LADDER))
    ap.add_argument("--levels", default="l0,l1,l2",
                    help="comma list of l0/l1/l2, or the literal 'md' for the "
                         "export + MD + L3/L4/L5 + report stage")
    ap.add_argument("--walltime", default=WALLTIME)
    ap.add_argument("--select", default=None,
                    help="override the PBS select line. Needed when the scheduler "
                         "keeps placing a job on a node whose GPUs are already "
                         "fully assigned: the job then sits at ~5%% CPU for hours "
                         "without failing, so pin it with ':host=<node>'.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    levels = [l.strip() for l in args.levels.split(",") if l.strip()]
    md_stage = levels == ["md"]
    if md_stage and args.walltime == WALLTIME:
        args.walltime = MD_WALLTIME
    for key in keys:
        entry = models.get(key)  # fail fast on a typo rather than at job start
        if md_stage:
            if entry.lammps_pair_in is None or not entry.lammps_pair_in.is_file():
                raise SystemExit(f"{key}: missing pair.in at {entry.lammps_pair_in}")
            # absolute path, not "bash <path>": pbs._render_commands only skips its
            # "python -u" prefix for commands starting with python, / or echo
            cmds = [f"{EXPORT} {key}"] + MD_STAGE
            tag = "md"
        else:
            cmds = [LEVELS[l] for l in levels]
            tag = "".join(levels)
        kw = {"select": args.select} if args.select else {}
        job = write_job(f"mev_{key}_{tag}", cmds, walltime=args.walltime,
                        model=key, **kw)
        if args.dry_run:
            print(f"[dry-run] {key}: {job.path}")
            continue
        jid = submit(job)
        print(f"{key:10s} -> {jid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
