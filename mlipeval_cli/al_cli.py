"""``alflow`` — command-line entry point of the active-learning scripts.

``alflow run`` drives the whole loop (committee training -> MD exploration ->
uncertainty selection -> DFT labelling -> merge -> next round) by forwarding to
``activate_learning_full_flow.py``.  The other subcommands expose one stage each,
built from the *same* functions of that script, so a platform scheduler can run
the stages as separate jobs and keep its own state between them::

    alflow train   ->  committee/model*/comm_v1_model*.model
    alflow sample  ->  <out>/ase_traj.xyz, ase_traj_with_uncertainty.xyz, uncertain_frames.xyz
    alflow label   ->  <root>/{0,1,...}/POSCAR + DFT template, optionally submitted
    alflow collect ->  <root>/dft_merged.extxyz
    alflow merge   ->  next training set
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

from . import _paths

CORE_NAME = "activate_learning_full_flow.py"


def _load_core(need_mace: bool):
    """Import the AL script as a module.

    Stages that do not touch MACE (label / collect) must also work on a login
    node without the MACE environment, so for those the two MACE imports are
    replaced by stubs that fail loudly if anything does call them.
    """
    path = _paths.alflow_home() / CORE_NAME
    if not need_mace:
        try:
            import mace.calculators  # noqa: F401
        except Exception:
            def _no_mace(*_a, **_k):
                raise RuntimeError("this alflow stage was started without MACE installed")
            for name in ("mace", "mace.calculators", "mace.cli", "mace.cli.run_train"):
                sys.modules.setdefault(name, types.ModuleType(name))
            sys.modules["mace.calculators"].MACECalculator = _no_mace
            sys.modules["mace.cli.run_train"].main = _no_mace
    spec = importlib.util.spec_from_file_location("alflow_core", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["alflow_core"] = mod  # dataclasses in the script look their module up here
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# --------------------------------------------------------------------------- config
#: alflow.dft key -> environment variable read by std.run / submitjobs.sh / sbatch
DFT_ENV = {
    "partition": "SBATCH_PARTITION",
    "account": "SBATCH_ACCOUNT",
    "modules": "VASP_MODULES",
    "vasp_cmd": "VASP_CMD",
    "max_jobs": "ALFLOW_MAX_JOBS",
}


def _read_config(path: str | None) -> dict:
    """The ``alflow`` section of a combined mlip_auto_flow.yaml (or a flat file
    whose keys are the run options, as in examples/alflow.yaml)."""
    import os

    if path is None:
        path = os.environ.get("ALFLOW_CONFIG") or os.environ.get("MLIPEVAL_CONFIG")
        if path is None:
            for cand in (Path.cwd() / "mlip_auto_flow.yaml", _paths.RELEASE_ROOT / "mlip_auto_flow.yaml"):
                if cand.exists():
                    path = str(cand)
                    break
    if not path:
        return {}
    p = Path(os.path.expandvars(path)).expanduser()
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text()) or {}
    else:
        import yaml
        data = yaml.safe_load(p.read_text()) or {}
    if "alflow" in data:
        return data["alflow"] or {}
    return {"run": data} if not ({"run", "dft"} & set(data)) else data


def _apply_dft_env(cfg: dict) -> dict:
    """Export alflow.dft settings; variables already set in the shell win."""
    import os

    applied = {}
    for key, env in DFT_ENV.items():
        val = (cfg.get("dft") or {}).get(key)
        if val is not None and not os.environ.get(env):
            os.environ[env] = str(val)
            applied[env] = str(val)
    return applied


def _run_option(cfg: dict, key: str):
    import os

    val = (cfg.get("run") or {}).get(key)
    if key == "foundation_model":
        val = val or os.environ.get("ALFLOW_FOUNDATION_MODEL")
    return val


# --------------------------------------------------------------------------- run
def _options_to_flags(opts: dict) -> list[str]:
    flags: list[str] = []
    for key, val in opts.items():
        flag = "--" + key
        if isinstance(val, bool):
            if val:
                flags.append(flag)
        elif isinstance(val, (list, tuple)):
            flags += [flag, *map(str, val)]
        elif val is not None:
            flags += [flag, str(val)]
    return flags


def cmd_run(ns, extra) -> int:
    cfg = _read_config(ns.config)
    _apply_dft_env(cfg)
    opts = dict(cfg.get("run") or {})
    home = _paths.alflow_home()
    given = {a.split("=")[0] for a in extra if a.startswith("--")}
    # the shell helpers ship with the scripts; use those unless told otherwise
    for key, name in (("submitjobs", "submitjobs.sh"), ("outcar2xyz", "outcar2xyz.sh")):
        if key not in opts and f"--{key}" not in given:
            opts[key] = str(home / name)
    flags = _options_to_flags(opts)
    cmd = [sys.executable, str(home / CORE_NAME), *flags, *extra]
    print("+", " ".join(cmd[1:]), file=sys.stderr)
    return subprocess.call(cmd)


# --------------------------------------------------------------------------- stages
def cmd_train(ns, extra) -> int:
    cfg = _read_config(ns.config)
    ns.foundation_model = ns.foundation_model or _run_option(cfg, "foundation_model")
    if not ns.foundation_model:
        sys.exit("alflow train: give --foundation-model, alflow.run.foundation_model, or $ALFLOW_FOUNDATION_MODEL")
    core = _load_core(need_mace=True)
    cfgs = core.write_committee_configs(
        train_file=Path(ns.train), valid_file=Path(ns.valid), test_file=Path(ns.test),
        device=ns.device, n_comm=ns.n_comm, comm_seed=ns.seeds[: ns.n_comm],
        foundation_model=Path(ns.foundation_model), config_dir=Path(ns.config_dir),
        committee_dir=Path(ns.committee_dir), comm_epoch=ns.epochs,
    )
    if ns.configs_only:
        print("\n".join(map(str, cfgs)))
        return 0
    core.train_committees(cfgs)
    models = [Path(ns.committee_dir) / f"model{i + 1}" / f"comm_v1_model{i + 1}.model" for i in range(ns.n_comm)]
    print(json.dumps({"models": [str(m) for m in models]}, indent=2))
    return 0


def cmd_sample(ns, extra) -> int:
    core = _load_core(need_mace=True)
    out = Path(ns.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    models = [Path(m) for m in ns.models]
    traj = out / "ase_traj.xyz"
    if ns.traj:
        traj = Path(ns.traj)
    else:
        core.run_ase_md(model_paths=models, init_conf_path=Path(ns.init_conf), md_temp=ns.temp,
                        md_steps=ns.steps, traj_out=traj, interval=ns.interval, device=ns.device)
    n = core.select_uncertain_frames(
        model_paths=models, traj_in=traj,
        traj_out_with_uncert=out / "ase_traj_with_uncertainty.xyz",
        uncertain_out=out / "uncertain_frames.xyz",
        threshold=ns.threshold, max_frames=ns.max_frames, device=ns.device,
    )
    print(json.dumps({"trajectory": str(traj), "selected": n,
                      "frames": str(out / "uncertain_frames.xyz")}, indent=2))
    return 0


def cmd_label(ns, extra) -> int:
    _apply_dft_env(_read_config(ns.config))
    core = _load_core(need_mace=False)
    home = _paths.alflow_home()
    root = Path(ns.root)
    n = core.prepare_dft_inputs(
        uncertain_xyz=Path(ns.frames), dft_root=root, template_dir=Path(ns.template_dir),
        files_to_copy=list(ns.files),
        submitjobs_src=Path(ns.submitjobs) if ns.submitjobs else home / "submitjobs.sh",
        outcar2xyz_src=Path(ns.outcar2xyz) if ns.outcar2xyz else home / "outcar2xyz.sh",
    )
    state = {"prepared": n, "root": str(root)}
    if n and ns.submit:
        script = root / (Path(ns.submitjobs).name if ns.submitjobs else "submitjobs.sh")
        core.run_bash(script=script, cwd=root, log_path=root / "submitjobs.log")
        state["submitted"] = True
    if n and ns.wait:
        done, failed = core.wait_for_dft(root, poll_sec=ns.poll_sec, timeout_sec=int(ns.timeout_hr * 3600))
        state.update(done=len(done), failed=[str(f) for f in failed])
    print(json.dumps(state, indent=2))
    return 0


def cmd_collect(ns, extra) -> int:
    core = _load_core(need_mace=False)
    root = Path(ns.root)
    core.ensure_plain_outcars(root)
    out = root / ns.name if ns.out is None else Path(ns.out)
    merged = None if ns.python_only else core.try_run_outcar2xyz(root, read_dir=".")
    if merged is not None and Path(merged).exists():
        shutil.copy2(merged, out)
        how = "outcar2xyz.sh"
    else:
        out = core.python_merge_from_outcar(root, out)
        how = "python fallback"
    nonconv = root / "non_converged_files.txt"
    print(json.dumps({"merged": str(out), "via": how,
                      "non_converged_list": str(nonconv) if nonconv.exists() else None}, indent=2))
    return 0


def cmd_merge(ns, extra) -> int:
    core = _load_core(need_mace=False)
    out = core.append_extxyz_concat(Path(ns.base), Path(ns.new), Path(ns.out))
    print(json.dumps({"train_set": str(out)}, indent=2))
    return 0


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="alflow", description="Uncertainty-driven active learning for MLIPs (MACE committee + VASP)")
    p.add_argument("--config", default=None,
                   help="mlip_auto_flow.yaml (its alflow: section) or a flat YAML/JSON of run options; "
                        "default: $ALFLOW_CONFIG, $MLIPEVAL_CONFIG, ./mlip_auto_flow.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("run", help="full AL loop; extra options go to activate_learning_full_flow.py")
    s.set_defaults(fn=cmd_run)
    s = sub.add_parser("config", help="show the alflow settings that would be used")
    s.set_defaults(fn=cmd_config)

    s = sub.add_parser("train", help="write committee configs and fine-tune N MACE models")
    s.add_argument("--train", required=True)
    s.add_argument("--valid", required=True)
    s.add_argument("--test", required=True)
    s.add_argument("--foundation-model", default=None, help="default: alflow.run.foundation_model or $ALFLOW_FOUNDATION_MODEL")
    s.add_argument("--n-comm", type=int, default=4)
    s.add_argument("--seeds", type=int, nargs="+", default=[123, 234, 345, 456])
    s.add_argument("--epochs", type=int, default=100)
    s.add_argument("--device", default="cuda")
    s.add_argument("--config-dir", default="config")
    s.add_argument("--committee-dir", default="committee")
    s.add_argument("--configs-only", action="store_true", help="write the YAML configs, do not train")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("sample", help="committee MD + uncertainty-based frame selection")
    s.add_argument("--models", nargs="+", required=True, help="committee .model files")
    s.add_argument("--init-conf", default=None, help="starting structure for MD")
    s.add_argument("--traj", default=None, help="skip MD and score an existing trajectory")
    s.add_argument("--temp", type=int, default=300)
    s.add_argument("--steps", type=int, default=1000)
    s.add_argument("--interval", type=int, default=10)
    s.add_argument("--threshold", type=float, default=0.9,
                   help="cut on min-max normalised sigma_E / sigma_F (0-1, relative to this trajectory)")
    s.add_argument("--max-frames", type=int, default=200)
    s.add_argument("--device", default="cuda")
    s.add_argument("--out-dir", default="comm_traj")
    s.set_defaults(fn=cmd_sample)

    s = sub.add_parser("label", help="write one VASP folder per selected frame; optionally submit and wait")
    s.add_argument("--frames", required=True, help="extxyz from `alflow sample`")
    s.add_argument("--template-dir", required=True, help="folder with INCAR, POTCAR, job script")
    s.add_argument("--files", nargs="+", default=["INCAR", "POTCAR", "std.run"])
    s.add_argument("--root", default="dft_cal")
    s.add_argument("--submitjobs", default=None)
    s.add_argument("--outcar2xyz", default=None)
    s.add_argument("--submit", action="store_true")
    s.add_argument("--wait", action="store_true")
    s.add_argument("--poll-sec", type=int, default=300)
    s.add_argument("--timeout-hr", type=float, default=24.0)
    s.set_defaults(fn=cmd_label)

    s = sub.add_parser("collect", help="merge finished OUTCARs into one extxyz")
    s.add_argument("--root", default="dft_cal")
    s.add_argument("--name", default="dft_merged.extxyz")
    s.add_argument("--out", default=None)
    s.add_argument("--python-only", action="store_true", help="skip outcar2xyz.sh, use the Python merger")
    s.set_defaults(fn=cmd_collect)

    s = sub.add_parser("merge", help="append newly labelled frames to the training set")
    s.add_argument("--base", required=True)
    s.add_argument("--new", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_merge)
    # `alflow run --config X` as well as `alflow --config X run`
    for sp in sub.choices.values():
        sp.add_argument("--config", dest="sub_config", default=None, help=argparse.SUPPRESS)
    return p


def cmd_config(ns, extra) -> int:
    cfg = _read_config(ns.config)
    env = {k: v for k, v in DFT_ENV.items()}
    print(json.dumps({"alflow_home": str(_paths.alflow_home()),
                      "run": cfg.get("run") or {},
                      "dft_env": {env[k]: (cfg.get("dft") or {}).get(k) for k in env}}, indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ns, extra = parser.parse_known_args(argv)
    ns.config = getattr(ns, "sub_config", None) or ns.config
    if extra and ns.cmd != "run":
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    if ns.cmd == "sample" and not (ns.traj or ns.init_conf):
        parser.error("sample needs --init-conf (run MD) or --traj (score an existing trajectory)")
    if extra and extra[0] == "--":
        extra = extra[1:]
    return int(ns.fn(ns, extra) or 0)


if __name__ == "__main__":
    sys.exit(main())
