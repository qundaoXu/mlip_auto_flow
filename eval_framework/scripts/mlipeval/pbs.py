"""PBS submission and the login-node guard.

Project rule (``CLAUDE.md``): every MACE / MD task must run as a PBS job on a
compute node, never on the login node.  This module makes that rule enforceable
rather than advisory:

* :func:`require_compute_node` aborts a heavy driver that was started on the
  login node and prints the exact ``qsub`` command that should have been used;
* :func:`select_device` picks CUDA when a GPU is actually visible, so a job that
  did land on a GPU node uses it;
* :func:`write_job` renders a job script from the template below, filled from
  the ``scheduler`` and ``environment`` sections of the site config
  (:mod:`mlipeval.config`), and :func:`submit` hands it to ``qsub``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from . import config
from .paths import OUTPUT, SCRIPTS

#: PBS project code (``#PBS -P``): ``scheduler.project`` / ``$MLIPEVAL_PBS_PROJECT``.
#: Unset -> the ``-P`` line is left out of the job script.
PROJECT = config.get("scheduler.project")

#: Default chunk: one GPU is what a MACE evaluation or MD run needs.
#:
#: The cpu and memory figures are the cluster's per-GPU quota rather than a
#: preference.  A site hook silently rewrites the chunk to 12 ncpus / 225 gb per
#: GPU, so asking for ``ncpus=10:mem=80gb`` does not get 10 cpus and 80 gb -- it
#: gets 12 and 225 while the job script claims otherwise, and ``ompthreads``
#: then leaves two cores idle.  Stating the enforced values keeps the script and
#: the running job describing the same thing.  ``ngpus=1`` also matters for
#: routing: every queue reachable through ``auto`` requires at least one GPU,
#: and 1-2 GPUs is what selects ``small`` (max 3 running jobs per user).
DEFAULT_SELECT = config.get("scheduler.select")
#: Default walltime for job scripts (``scheduler.walltime``).
DEFAULT_WALLTIME = config.get("scheduler.walltime")
#: Default queue (``scheduler.queue``); ``None`` lets PBS route the job.
DEFAULT_QUEUE = config.get("scheduler.queue")

JOB_TEMPLATE = """#!/bin/bash
#PBS -N {name}
{project_line}{queue_line}#PBS -l walltime={walltime}
#PBS -l select={select}
#PBS -j oe
#PBS -o {logfile}
cd {workdir}
{module_lines}source {conda_sh}
conda activate {conda_env}
{model_line}{engine_line}export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$NCPUS
export MLIPEVAL_THREADS=$NCPUS

echo "=== node: $(hostname)  job: $PBS_JOBID  $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "no nvidia-smi"

{commands}

RC=$?
echo "=== done rc=$RC $(date) ==="
exit $RC
"""


def in_pbs_job() -> bool:
    """True when running inside a PBS job rather than on the login node."""
    return bool(os.environ.get("PBS_JOBID"))


def gpu_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def select_device(prefer: str = "auto") -> str:
    """``"cuda"`` when a GPU is visible, otherwise ``"cpu"``."""
    if prefer in ("cuda", "cpu"):
        return prefer
    return "cuda" if gpu_available() else "cpu"


def describe_environment() -> dict[str, object]:
    return {
        "in_pbs_job": in_pbs_job(),
        "pbs_jobid": os.environ.get("PBS_JOBID"),
        "hostname": os.uname().nodename,
        "gpu_available": gpu_available(),
        "device": select_device(),
        "threads": os.environ.get("MLIPEVAL_THREADS") or os.environ.get("NCPUS"),
    }


def require_compute_node(task: str, *, allow_login_node: bool = False) -> None:
    """Abort a MACE-heavy driver that was launched on the login node.

    ``allow_login_node`` exists only for smoke tests small enough not to matter;
    it has to be passed explicitly, so the default behaviour cannot drift back
    to running production work on the head node.
    """
    if allow_login_node or in_pbs_job():
        return
    flag = config.get("scheduler.require_compute_node", True)
    if str(flag).lower() in ("false", "0", "no", "off"):
        return  # site config: no batch system, run heavy drivers in place
    invocation = " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]).strip()
    print(
        f"拒绝在登录节点上运行 MACE 任务：{task}\n"
        f"\n项目规定（CLAUDE.md）所有 MACE/MD 计算都必须提交 PBS。请改用：\n"
        f"\n    cd {SCRIPTS}\n"
        f'    python submit_pbs.py --command "{invocation}"\n'
        f"\n或提交整条链：\n"
        f"\n    python submit_pbs.py --chain\n"
        f"\n只在确实无关紧要的小规模冒烟测试时，才加 --allow-login-node 绕过本检查。",
        file=sys.stderr,
    )
    raise SystemExit(3)


@dataclass
class Job:
    name: str
    path: Path
    logfile: Path
    commands: list[str]


def _render_commands(commands: list[str]) -> str:
    """Render driver invocations as a fail-fast sequence.

    The naive rendering (one command per line) is wrong for any job holding more
    than one of them, and the whole chain-in-one-job mode is nothing but that:
    the template's trailing ``RC=$?`` sees only the *last* command, so a failure
    anywhere earlier neither stops the rest nor reaches the exit code.  A job
    whose MD seed 0 died would still run the analysis over the missing
    trajectory and still report success to ``afterok``.

    So each step is guarded and the first non-zero exit ends the job with that
    code, which is exactly what the multi-job chain's ``afterok`` edges express.
    ``set -e`` would not do: these are pipelines under ``python -u``, and a bare
    ``set -e`` says nothing about which step died.
    """
    n = len(commands)
    lines: list[str] = []
    for i, c in enumerate(commands, 1):
        cmd = f"python -u {c}" if not c.startswith(("python", "/", "echo")) else c
        label = c.replace('"', r"\"")
        lines.append(f'echo "=== step {i}/{n}: {label}  $(date -Is) ==="')
        lines.append(cmd)
        lines.append(
            f'rc=$?; if [ $rc -ne 0 ]; then '
            f'echo "STEP_FAILED step={i}/{n} rc=$rc cmd={label}"; exit $rc; fi'
        )
    return "\n".join(lines)


def write_job(
    name: str,
    commands: list[str] | str,
    *,
    directory: Path | None = None,
    walltime: str | None = None,
    select: str = DEFAULT_SELECT,
    queue: str | None = None,
    workdir: Path | None = None,
    model: str | None = None,
) -> Job:
    """Render a PBS job script for one or more driver invocations.

    ``queue`` is left unset by default: this cluster routes through ``auto`` and
    the group's working job scripts specify no queue at all, so naming one
    (there is no ``normal`` queue here) would only cause the submission to be
    rejected.

    ``model`` selects one of :data:`mlipeval.models.REGISTRY`; the job then
    activates *that model's* conda environment and exports ``MLIPEVAL_MODEL``,
    which is what routes its output into ``results/models/<key>/``.  The five
    backends cannot share one environment (conflicting e3nn and torch pins), so
    this is the only place the mapping is applied.
    """
    from .models import get as get_model

    entry = get_model(model) if model else None
    conda_env = entry.conda_env if entry else config.backend_env("mace")
    model_line = f"export MLIPEVAL_MODEL={entry.key}\n" if entry else ""
    # 引擎必须随作业一起钉住：链里的 run_mlip_md 写、L3/L4/L5 读，
    # 两边都从 MLIPEVAL_MD_ENGINE 取，不导出就会各用各的默认值。
    from .state_points import current_engine
    engine_line = f"export MLIPEVAL_MD_ENGINE={current_engine()}\n"
    directory = Path(directory) if directory else SCRIPTS / "pbs"
    directory.mkdir(parents=True, exist_ok=True)
    workdir = Path(workdir) if workdir else SCRIPTS
    if isinstance(commands, str):
        commands = [commands]
    walltime = walltime or DEFAULT_WALLTIME
    queue = queue or DEFAULT_QUEUE
    modules = config.get("scheduler.modules") or []
    if isinstance(modules, str):
        modules = modules.split(",")
    module_lines = ("source /etc/profile.d/modules.sh\n" + "".join(f"module load {m.strip()}\n" for m in modules)) if modules else ""
    conda_sh = config.conda_root() / "etc/profile.d/conda.sh"

    logfile = (OUTPUT / "results" / "pbs" / f"{name}.log")
    logfile.parent.mkdir(parents=True, exist_ok=True)
    body = _render_commands(commands)
    path = directory / f"{name}.pbs"
    path.write_text(
        JOB_TEMPLATE.format(
            name=name, project_line=(f"#PBS -P {PROJECT}\n" if PROJECT else ""),
            module_lines=module_lines, conda_sh=conda_sh,
            queue_line=(f"#PBS -q {queue}\n" if queue else ""), walltime=walltime,
            select=select, logfile=logfile, workdir=workdir, commands=body,
            conda_env=conda_env, model_line=model_line, engine_line=engine_line,
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)
    return Job(name=name, path=path, logfile=logfile, commands=commands)


def submit(job: Job, *, depends_on: str | None = None, dry_run: bool = False) -> str | None:
    """``qsub`` the job; returns the job id, or ``None`` for a dry run."""
    if shutil.which("qsub") is None:
        raise RuntimeError("qsub not found -- are you on the cluster?")
    cmd = ["qsub"]
    if depends_on:
        cmd += ["-W", f"depend=afterok:{depends_on}"]
    cmd.append(str(job.path))
    if dry_run:
        print("(dry run) " + " ".join(cmd))
        return None
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=str(job.path.parent))
    if out.returncode != 0:
        raise RuntimeError(f"qsub failed: {out.stderr.strip() or out.stdout.strip()}")
    return out.stdout.strip()
