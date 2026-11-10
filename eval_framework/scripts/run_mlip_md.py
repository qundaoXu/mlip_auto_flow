#!/usr/bin/env python
"""Driver: MLIP MD runs matched to the AIMD reference state points.

    python run_mlip_md.py --state canon_amorphous_300K [--steps N] [--seed K] [--threads T]
    python run_mlip_md.py --state all --seed K

Each state point starts from the AIMD trajectory's own first frame, in the AIMD
cell, at the AIMD temperature and timestep.  For the canonical state points seed
``K`` starts from AIMD replica ``K + 1``, so seeds 0-2 mirror the three replicas
one to one.  ``--state all`` runs the active (canonical) state points only;
legacy ones are still selectable by name.  The only thing that differs between
the two sides is the engine, which is the precondition for L3/L4/L5 comparisons
to mean anything.

Trajectories are written to ``eval/data/md_<state>.npz`` and reloaded by the
L3/L4/L5 drivers, so the expensive part is paid once.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


from mlipeval import paths
from mlipeval.calculators import build_model, stdout_progress
from mlipeval.pbs import describe_environment, require_compute_node
from mlipeval.md_runner import MDSettings, run_md, save_md
from mlipeval.state_points import (
    CANONICAL_STATES, DEFAULT_MD_ENGINE, MD_ENGINES, STATES, StatePoint,
    current_engine, get_state, reference_trajectory,
)


def run_state(state: StatePoint, model, *, steps: int | None, seed: int, threads: int | None,
              engine: str = "lammps") -> Path:
    replica = state.replica_for_seed(seed)
    reference = reference_trajectory(state, replica=replica, max_frames=3)
    start = reference.atoms(0)
    n_steps = steps or state.default_steps

    settings = MDSettings(
        ensemble="nvt",
        thermostat="nose-hoover-chain",
        temperature_k=state.temperature_k,
        temperature_end_k=state.temperature_end_k,
        timestep_fs=3.0,
        n_steps=n_steps,
        sample_every=state.sample_every,
        seed=seed,
        engine=engine,
        equilibration_fraction=state.equilibration_fraction,
        protocol_id=f"mlip_md_{state.name}_seed{seed}",
        description=(
            f"MLIP MD mirroring {state.reference} replica {replica}: {len(start)} atoms, "
            f"V={start.get_volume():.2f} A^3, rho={len(start) / start.get_volume():.5f} at/A^3"
        ),
    )
    print(
        f"\n=== {state.name} ===\n"
        f"  reference : {state.reference}, replica {replica} ({state.note})\n"
        f"  start     : {start.get_chemical_formula()}, {len(start)} atoms, "
        f"rho={len(start) / start.get_volume():.5f} at/A^3\n"
        f"  programme : {settings.ensemble.upper()} "
        f"{settings.temperature_k:.0f} K"
        + (f" -> {settings.temperature_end_k:.0f} K" if settings.temperature_end_k else "")
        + f", dt={settings.timestep_fs} fs, {n_steps} steps "
        f"({n_steps * settings.timestep_fs / 1000:.2f} ps)"
    )

    if engine == "lammps":
        from mlipeval import models as model_registry
        from mlipeval.md_lammps import run_md_lammps

        entry = model_registry.current()
        workdir = paths.data_path(f"lammps_md/{state.name}_seed{seed}")
        print(f"  engine    : LAMMPS ({entry.lammps_route}), workdir {workdir}")
        result = run_md_lammps(start, model, settings, entry=entry, workdir=workdir,
                               progress=stdout_progress("  MD "))
    else:
        print("  engine    : ASE")
        result = run_md(start, model, settings, progress=stdout_progress("  MD "), progress_every=25)
    # 写路径与读路径必须是同一个函数：早先这里自己拼文件名，而 L3/L4/L5 走
    # StatePoint.md_path_for()，两边一改就对不上（引擎进文件名时正是这样炸的）。
    path = save_md(result, state.md_path_for(seed, engine))
    print(f"  {json.dumps(result.summary())}")
    print(f"  wrote {path}")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", default="all", choices=[*STATES, "all"])
    ap.add_argument("--steps", type=int, default=None, help="override the step count")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--allow-login-node", action="store_true",
                    help="bypass the compute-node check; only for trivial smoke tests")
    ap.add_argument("--engine", default=None, choices=list(MD_ENGINES),
                    help="MD engine; defaults to MLIPEVAL_MD_ENGINE (else "
                         f"{DEFAULT_MD_ENGINE}). 两个引擎物理等价（见 "
                         "scripts/experiments/lammps_md/README.md），选择依据是 "
                         "lammps=生产引擎、ase=单一代码路径且对 MACE/DPA 更快。"
                         "**跨模型比较必须固定同一个引擎。**")
    args = ap.parse_args()
    # 写与读走同一个开关：L3/L4/L5 是从 MLIPEVAL_MD_ENGINE 取引擎来定位轨迹的，
    # 只在这里认 --engine 会让同一条链写进 A 的文件名、又去读 B。
    if args.engine:
        os.environ["MLIPEVAL_MD_ENGINE"] = args.engine
    engine = current_engine()
    require_compute_node('MLIP molecular dynamics', allow_login_node=args.allow_login_node)

    if engine == "lammps":
        # LAMMPS 跑在子进程里，力由它自己的 pair_style 算——这里再建一个 ASE
        # calculator 只会白占一份显存，所以只取模型身份（spec 进 protocol / 溯源）。
        from mlipeval.calculators import Model, model_spec_only

        model = Model(calculator=None, spec=model_spec_only(device=args.device))
    else:
        model = build_model(threads=args.threads, device=args.device)
    print(f"model: {model.spec.name} sha256={model.spec.checkpoint_sha256[:16]} dtype={model.spec.dtype}")

    names = list(CANONICAL_STATES) if args.state == "all" else [args.state]
    for name in names:
        run_state(get_state(name), model, steps=args.steps, seed=args.seed, threads=args.threads,
                  engine=engine)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
