#!/usr/bin/env python3
"""
Active-learning workflow skeleton (MACE committee + ASE MD + DFT labeling).

What this script adds vs your current skeleton:
1) Run submitjobs.sh under dft_cal/ to submit the DFT batch.
2) Poll the DFT folders until calculations are finished (based on OUTCAR tail).
3) Run outcar2xyz.sh to merge results into a single extxyz (with fallback merger in Python).
4) Append the merged DFT-labeled extxyz to the current training set and start the next AL round.

Assumptions:
- dft_cal/{0,1,2,...}/ contain a VASP run (OUTCAR produced).
- dft_cal/submitjobs.sh submits all jobs under dft_cal/*/ (you already copy it there).
- dft_cal/outcar2xyz.sh merges all finished calculations into ONE xyz/extxyz (preferred).
  If that script does not produce a merged xyz file, we fallback to Python merging from OUTCAR.

Usage example:
  python activate_learning_flow_full.py --n_rounds 3 --md_temp 600 --md_steps 5000
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from ase.io import read, write
from tqdm import tqdm

from mace.calculators import MACECalculator
from mace.cli.run_train import main as mace_run_train_main
from sklearn.preprocessing import MinMaxScaler

# Optional dependency (your current script uses it)
try:
    from aseMolec import extAtoms as ea  # type: ignore
except Exception:
    ea = None

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# ------------------------- logging ------------------------- #
def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("active_learning")
    logger.handlers.clear()
    logger.propagate = False

    log_file = Path.cwd() / f"active_learning_{int(time.time())}.log"
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    # also print to console
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info("Log file: %s", log_file)
    return logger


logger = setup_logger()


# ------------------------- helpers ------------------------- #
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run_bash(script: Path, cwd: Path, args: Optional[List[str]] = None, log_path: Optional[Path] = None) -> None:
    if not script.exists():
        raise FileNotFoundError(f"Missing script: {script}")
    cmd = ["bash", script.name] + (args or [])
    logger.info("Running: %s (cwd=%s)", " ".join(cmd), cwd)

    stdout = subprocess.PIPE
    stderr = subprocess.STDOUT
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=stdout, stderr=stderr, text=True)

    lines: List[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line)
        if len(lines) <= 2000:  # avoid huge memory
            logger.info("[bash] %s", line.rstrip())

    ret = proc.wait()
    if log_path is not None:
        ensure_dir(log_path.parent)
        log_path.write_text("".join(lines), encoding="utf-8")
    if ret != 0:
        raise RuntimeError(f"{script} failed with return code {ret}")


def list_numeric_subdirs(root: Path) -> List[Path]:
    sub = []
    for p in root.iterdir():
        if p.is_dir() and re.fullmatch(r"\d+", p.name):
            sub.append(p)
    return sorted(sub, key=lambda x: int(x.name))


def tail_text(path: Path, n_bytes: int = 20000) -> str:
    """Read last ~n_bytes of a text file. Supports plain text and .gz."""
    if path.exists():
        with path.open("rb") as f:
            try:
                f.seek(-n_bytes, os.SEEK_END)
            except OSError:
                f.seek(0)
            data = f.read()
        return data.decode("utf-8", errors="ignore")

    gz = Path(str(path) + ".gz")
    if gz.exists():
        # gzip doesn't support efficient tail; stream and keep last n_bytes
        buf = b""
        with gzip.open(gz, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                buf = (buf + chunk)[-n_bytes:]
        return buf.decode("utf-8", errors="ignore")

    return ""


@dataclass
class OutcarStatus:
    done: bool
    failed: bool
    reason: str = ""


DONE_PATTERNS = [
    "General timing and accounting informations for this job",
    "Voluntary context switches",
    "reached required accuracy",
    "Total CPU time used",
]
FAIL_PATTERNS = [
    "ZBRENT: fatal error",
    "BRMIX: very serious problems",
    "ERROR",
    "FATAL",
    "internal error",
]


def check_outcar_status(outcar: Path) -> OutcarStatus:
    txt = tail_text(outcar, n_bytes=30000)
    if txt.strip() == "":
        return OutcarStatus(done=False, failed=False, reason="OUTCAR(.gz) missing or empty")

    # failure (try to be conservative)
    for fp in FAIL_PATTERNS:
        if fp in txt:
            return OutcarStatus(done=False, failed=True, reason=f"Found failure pattern: {fp}")

    for dp in DONE_PATTERNS:
        if dp in txt:
            return OutcarStatus(done=True, failed=False, reason=f"Found done pattern: {dp}")

    return OutcarStatus(done=False, failed=False, reason="Not finished yet")


def wait_for_dft(root: Path, poll_sec: int, timeout_sec: int) -> Tuple[List[Path], List[Path]]:
    """
    Return (done_dirs, failed_dirs). Raise on timeout (but prints progress).
    """
    start = time.time()
    calc_dirs = list_numeric_subdirs(root)
    if not calc_dirs:
        raise RuntimeError(f"No numeric subfolders found under {root}")

    logger.info("Waiting DFT: %d calculations under %s", len(calc_dirs), root)

    done_dirs: List[Path] = []
    failed_dirs: List[Path] = []

    while True:
        done_dirs.clear()
        failed_dirs.clear()
        running_dirs: List[Path] = []

        for d in calc_dirs:
            outcar = d / "OUTCAR"
            st = check_outcar_status(outcar)
            if st.done:
                done_dirs.append(d)
            elif st.failed:
                failed_dirs.append(d)
            else:
                running_dirs.append(d)

        logger.info("DFT status: done=%d, running=%d, failed=%d",
                    len(done_dirs), len(running_dirs), len(failed_dirs))

        if len(done_dirs) + len(failed_dirs) == len(calc_dirs):
            break

        if time.time() - start > timeout_sec:
            raise TimeoutError(
                f"Timeout waiting DFT. done={len(done_dirs)} running={len(running_dirs)} failed={len(failed_dirs)}"
            )

        time.sleep(poll_sec)

    return done_dirs, failed_dirs

def ensure_plain_outcars(dft_root: Path) -> None:
    """
    outcar2xyz.sh only searches for files named 'OUTCAR' (not OUTCAR.gz).
    If you have OUTCAR.gz, we create a plain OUTCAR copy for conversion.
    """
    for d in list_numeric_subdirs(dft_root):
        outcar = d / "OUTCAR"
        outcar_gz = d / "OUTCAR.gz"
        if (not outcar.exists()) and outcar_gz.exists():
            try:
                import gzip
                with gzip.open(outcar_gz, "rb") as fin, open(outcar, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
                logger.info("Decompressed %s -> %s for conversion", outcar_gz, outcar)
            except Exception as e:
                logger.warning("Failed to decompress %s: %s", outcar_gz, e)


# ------------------------- MACE training ------------------------- #
def write_committee_configs(
    train_file: Path,
    valid_file: Path,
    test_file: Path,
    device: str,
    n_comm: int,
    comm_seed: List[int],
    foundation_model: Path,
    config_dir: Path,
    committee_dir: Path,
    comm_epoch: int,
) -> List[Path]:
    ensure_dir(config_dir)
    ensure_dir(committee_dir)

    cfg_paths: List[Path] = []
    for i in range(n_comm):
        model_dir = committee_dir / f"model{i + 1}"
        ensure_dir(model_dir)

        config_text = (
            f'name: "comm_v1_model{i + 1}"\n'
            'multiheads_finetuning: False\n'
            f'train_file: "{train_file.as_posix()}"\n'
            f'valid_file: "{valid_file.as_posix()}"\n'
            f'test_file: "{test_file.as_posix()}"\n'
            f'work_dir: "{model_dir.as_posix()}"\n'
            f'foundation_model: "{foundation_model.as_posix()}"\n'
            'E0s: "foundation"\n'
            'energy_weight: 10.0\n'
            'forces_weight: 10.0\n'
            'stress_weight: 1.0\n'
            'energy_key: "REF_energy"\n'
            'forces_key: "REF_forces"\n'
            'stress_key: "REF_stress"\n'
            'virials_key: "Virial"\n'
            'compute_stress: True\n'
            'batch_size: 4\n'
            f'device: {device}\n'
            f'max_num_epochs: {comm_epoch}\n'
            f'seed: {comm_seed[i]}\n'
            'enable_cueq: True\n'
        )

        config_path = config_dir / f"config-0{i + 1}.yml"
        config_path.write_text(config_text, encoding="utf-8")
        cfg_paths.append(config_path)
        logger.info("Saved %s", config_path)

    return cfg_paths


def train_mace(config_file: Path) -> None:
    # MACE CLI reads sys.argv
    sys.argv = ["program", "--config", str(config_file)]
    mace_run_train_main()


def train_committees(config_paths: List[Path]) -> None:
    logger.info("Start training committees (%d models)", len(config_paths))
    for cfg in config_paths:
        logger.info("Training with config: %s", cfg)
        train_mace(cfg)


# ------------------------- MD sampling ------------------------- #
def run_ase_md(
    model_paths: List[Path],
    init_conf_path: Path,
    md_temp: int,
    md_steps: int,
    traj_out: Path,
    interval: int = 10,
) -> None:
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import Stationary, ZeroRotation, MaxwellBoltzmannDistribution

    ensure_dir(traj_out.parent)
    if traj_out.exists():
        traj_out.unlink()

    mace_calc = MACECalculator(model_paths=[p.as_posix() for p in model_paths], device="cuda", default_dtype="float64")

    init_conf = read(init_conf_path.as_posix(), index=0).copy()
    init_conf.calc = mace_calc

    np.random.seed(42)
    MaxwellBoltzmannDistribution(init_conf, temperature_K=md_temp)
    Stationary(init_conf)
    ZeroRotation(init_conf)

    dyn = Langevin(init_conf, 1 * units.fs, temperature_K=md_temp, friction=0.1)

    def write_frame():
        at = dyn.atoms.copy()
        write(traj_out.as_posix(), at, format="extxyz", append=True, write_results=False)

    dyn.attach(write_frame, interval=interval)
    dyn.run(md_steps)

    logger.info("MD finished. Trajectory written to %s", traj_out)


# ------------------------- uncertainty selection ------------------------- #
def select_uncertain_frames(
    model_paths: List[Path],
    traj_in: Path,
    traj_out_with_uncert: Path,
    uncertain_out: Path,
    threshold: float = 0.9,
    max_frames: Optional[int] = None,
) -> int:
    ensure_dir(traj_out_with_uncert.parent)
    ensure_dir(uncertain_out.parent)

    val_traj = read(traj_in.as_posix(), index=":")
    logger.info("Loaded %d MD frames from %s", len(val_traj), traj_in)

    mace_calc = MACECalculator(model_paths=[p.as_posix() for p in model_paths], device="cuda", default_dtype="float64")
    single_calcs = [MACECalculator(model_paths=[p.as_posix()], device="cuda", default_dtype="float64") for p in model_paths]

    glob_evar = []
    glob_fsig_rms = []
    glob_fsig_maxatom = []

    for at in tqdm(val_traj, desc="uncertainty"):
        at.calc = mace_calc
        engs = at.get_potential_energies()
        at.info["sigma_E"] = float(np.std(engs))
        at.info["average_mace_energy"] = float(np.average(engs))

        # record per-model energies for debug
        for j, e in enumerate(engs, start=1):
            at.info[f"energy_mace_{j}"] = float(e)

        # forces per model
        forces_stack = []
        for calc in single_calcs:
            at_single = at.copy()
            at_single.calc = calc
            forces_stack.append(at_single.get_forces())
        forces_stack = np.stack(forces_stack, axis=0)  # (n_comm, N, 3)
        f_std = np.std(forces_stack, axis=0, ddof=0)   # (N, 3)

        sigmaF_rms = float(np.sqrt(np.mean(f_std ** 2)))
        at.info["sigmaF_rms"] = sigmaF_rms
        glob_fsig_rms.append(sigmaF_rms)

        per_atom_sigma = np.sqrt(np.mean(f_std ** 2, axis=1))  # (N,)
        sigmaF_maxatom = float(np.max(per_atom_sigma))
        at.info["sigmaF_maxatom"] = sigmaF_maxatom
        glob_fsig_maxatom.append(sigmaF_maxatom)

        glob_evar.append(float(np.std(engs)))

    # clear calculators before writing
    for at in val_traj:
        at.calc = None

    write(traj_out_with_uncert.as_posix(), val_traj, format="extxyz")
    logger.info("Trajectory with uncertainty written to %s", traj_out_with_uncert)

    # Collect arrays
    if ea is not None:
        var_E = ea.get_prop(val_traj, "info", "sigma_E", peratom=False)
        var_F = ea.get_prop(val_traj, "info", "sigmaF_rms", peratom=False)
        var_nmaxF = ea.get_prop(val_traj, "info", "sigmaF_maxatom", peratom=False)
    else:
        var_E = np.array([a.info["sigma_E"] for a in val_traj], dtype=float)
        var_F = np.array([a.info["sigmaF_rms"] for a in val_traj], dtype=float)
        var_nmaxF = np.array([a.info["sigmaF_maxatom"] for a in val_traj], dtype=float)

    data = np.column_stack([var_E, var_F, var_nmaxF])
    scaler_mm = MinMaxScaler()
    data_mm = scaler_mm.fit_transform(data)
    var_E_mm, var_F_mm, var_nmaxF_mm = data_mm.T

    idx_e = np.where(var_E_mm > threshold)[0]
    idx_force = np.where(var_F_mm > threshold)[0]
    idx_maxforce = np.where(var_nmaxF_mm > threshold)[0]
    idx_union = np.sort(np.unique(np.concatenate([idx_e, idx_force, idx_maxforce])))

    if max_frames is not None and len(idx_union) > max_frames:
        # keep the highest combined score
        score = var_E_mm + var_F_mm + var_nmaxF_mm
        idx_union = idx_union[np.argsort(score[idx_union])[::-1][:max_frames]]
        idx_union = np.sort(idx_union)

    logger.info("Uncertain frames: %d (threshold=%.3f)", len(idx_union), threshold)
    write(uncertain_out.as_posix(), [val_traj[i].copy() for i in idx_union], format="extxyz")
    return int(len(idx_union))


# ------------------------- DFT I/O ------------------------- #
def prepare_dft_inputs(
    uncertain_xyz: Path,
    dft_root: Path,
    template_dir: Path,
    files_to_copy: List[str],
    submitjobs_src: Path,
    outcar2xyz_src: Path,
) -> int:
    ensure_dir(dft_root)

    # clean only numeric subdirs from previous round
    for d in list_numeric_subdirs(dft_root):
        shutil.rmtree(d, ignore_errors=True)

    # also remove old merged xyz in root
    for p in dft_root.glob("*.xyz"):
        try:
            p.unlink()
        except Exception:
            pass

    uncertain_traj = read(uncertain_xyz.as_posix(), index=":")
    if len(uncertain_traj) == 0:
        logger.warning("No uncertain frames in %s", uncertain_xyz)
        return 0

    for i, atoms in enumerate(uncertain_traj):
        d = dft_root / f"{i}"
        ensure_dir(d)
        write((d / "POSCAR").as_posix(), atoms, format="vasp", direct=True)

        for fname in files_to_copy:
            src = template_dir / fname
            if not src.exists():
                raise FileNotFoundError(f"Missing template file: {src}")
            shutil.copy2(src, d / fname)

    shutil.copy2(submitjobs_src, dft_root / submitjobs_src.name)
    shutil.copy2(outcar2xyz_src, dft_root / outcar2xyz_src.name)

    logger.info("Prepared %d DFT folders under %s", len(uncertain_traj), dft_root)
    return int(len(uncertain_traj))


def try_run_outcar2xyz(dft_root: Path, read_dir: str = ".") -> Optional[Path]:
    """
    Run outcar2xyz.sh (your converter) and return the merged extxyz path if found.

    Your outcar2xyz.sh requires ONE argument: the directory to search for OUTCAR files.
    It writes to:  <dft_root>/NEPdataset-multiple_frames/NEP-dataset.xyz
    """
    script = dft_root / "outcar2xyz.sh"
    if not script.exists():
        logger.warning("outcar2xyz.sh not found under %s", dft_root)
        return None

    # Run with required argument (e.g., "." to scan all numeric subdirs)
    t0 = time.time()
    run_bash(script=script, cwd=dft_root, args=[read_dir], log_path=dft_root / "outcar2xyz.log")

    # Preferred output (as defined in your script)
    preferred = dft_root / "NEPdataset-multiple_frames" / "NEP-dataset.xyz"
    if preferred.exists() and preferred.stat().st_size > 0:
        logger.info("Merged DFT extxyz from outcar2xyz.sh: %s", preferred)
        return preferred

    # Fallback: find newest xyz/extxyz anywhere under dft_root
    candidates = []
    for p in dft_root.rglob("*.xyz"):
        if p.is_file() and p.stat().st_mtime >= t0 - 2 and p.stat().st_size > 0:
            candidates.append(p)
    for p in dft_root.rglob("*.extxyz"):
        if p.is_file() and p.stat().st_mtime >= t0 - 2 and p.stat().st_size > 0:
            candidates.append(p)

    if not candidates:
        logger.warning("outcar2xyz.sh ran but no merged xyz/extxyz detected under %s", dft_root)
        return None

    newest = max(candidates, key=lambda x: x.stat().st_mtime)
    logger.info("Detected merged DFT file (fallback): %s", newest)
    return newest


def python_merge_from_outcar(dft_root: Path, merged_out: Path) -> Path:
    """
    Fallback merger: read each finished OUTCAR and write a merged extxyz with keys:
      REF_energy, REF_forces, REF_stress, Virial
    """
    calc_dirs = list_numeric_subdirs(dft_root)
    frames = []
    n_ok = 0
    n_fail = 0

    for d in calc_dirs:
        outcar = d / "OUTCAR"
        st = check_outcar_status(outcar)
        if not st.done:
            n_fail += 1
            continue
        try:
            at = read(outcar.as_posix(), index=-1)  # last ionic step
            # SinglePointCalculator attached by ASE should provide energy/forces/stress
            e = float(at.get_potential_energy())
            f = np.array(at.get_forces(), dtype=float)
            try:
                stress = np.array(at.get_stress(), dtype=float)  # voigt (6,)
            except Exception:
                stress = None

            at.calc = None  # avoid writing calc internals

            at.info["REF_energy"] = e
            at.arrays["REF_forces"] = f

            if stress is not None and stress.shape == (6,):
                at.info["REF_stress"] = stress.tolist()
                # Virial in Voigt: -stress * volume (same convention used in many ML datasets)
                vol = float(at.get_volume())
                virial = (-stress * vol).tolist()
                at.info["Virial"] = virial

            frames.append(at)
            n_ok += 1
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", outcar, exc)
            n_fail += 1

    if n_ok == 0:
        raise RuntimeError(f"Python merger found 0 finished OUTCARs under {dft_root}")

    ensure_dir(merged_out.parent)
    write(merged_out.as_posix(), frames, format="extxyz")
    logger.info("Python-merged DFT results: %d frames (skipped %d)", n_ok, n_fail)
    return merged_out


def append_extxyz_concat(base_xyz: Path, new_xyz: Path, out_xyz: Path) -> Path:
    """Fast concatenation for extxyz: byte-wise cat."""
    ensure_dir(out_xyz.parent)
    with base_xyz.open("rb") as f1, new_xyz.open("rb") as f2, out_xyz.open("wb") as fo:
        b1 = f1.read()
        b2 = f2.read()
        fo.write(b1)
        if len(b1) > 0 and not b1.endswith(b"\n"):
            fo.write(b"\n")
        fo.write(b2)
        if len(b2) > 0 and not b2.endswith(b"\n"):
            fo.write(b"\n")
    return out_xyz


# ------------------------- main loop ------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Active learning workflow (MACE committee + DFT labeling).")

    # dataset / model
    parser.add_argument("--data_path", type=str, default="data", help="Dataset folder")
    parser.add_argument("--foundation_model", type=str, default="/home/qdx/1-MACE/mace/6-AL_flow/mace-mpa-0-medium.model")
    parser.add_argument("--subset_size", type=int, default=500)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    # committee / MD
    parser.add_argument("--n_comm", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--comm_epoch", type=int, default=1)
    parser.add_argument("--comm_seed", type=int, nargs="+", default=[123, 234, 345, 456])
    parser.add_argument("--init_conf_path", type=str, default="data/small_cell_traj.xyz")
    parser.add_argument("--md_temp", type=int, default=300)
    parser.add_argument("--md_steps", type=int, default=1000)
    parser.add_argument("--md_write_interval", type=int, default=10)

    # selection
    parser.add_argument("--uncert_threshold", type=float, default=0.9)
    parser.add_argument("--uncert_max_frames", type=int, default=200)

    # AL loop
    parser.add_argument("--n_rounds", type=int, default=1)

    # DFT batch
    parser.add_argument("--dft_root", type=str, default="dft_cal")
    parser.add_argument("--dft_template_dir", type=str, default="dft_input")
    parser.add_argument("--dft_template_files", type=str, nargs="+", default=["INCAR", "POTCAR", "std.run"])
    parser.add_argument("--submitjobs", type=str, default="submitjobs.sh")
    parser.add_argument("--outcar2xyz", type=str, default="outcar2xyz.sh")
    parser.add_argument("--dft_poll_sec", type=int, default=300)
    parser.add_argument("--dft_timeout_hr", type=float, default=24.0)
    parser.add_argument("--merged_dft_name", type=str, default="dft_merged.extxyz")

    # outputs
    parser.add_argument("--work_dir", type=str, default=".")
    parser.add_argument("--keep_all_rounds", action="store_true", help="Keep per-round outputs (no overwrite)")
    parser.add_argument("--force_resample", action="store_true", help="Resample init_dataset.xyz even if sub_train exists")

    args = parser.parse_args([]) if "get_ipython" in globals() else parser.parse_args()

    work_dir = Path(args.work_dir).resolve()
    os.chdir(work_dir)

    data_dir = Path(args.data_path)
    ensure_dir(data_dir)
    ensure_dir(Path("config"))
    ensure_dir(Path("committee"))
    ensure_dir(Path("comm_traj"))

    # ---------- 0) initial subsample / split (only once) ---------- #
    sub_train = data_dir / "sub_train.xyz"
    sub_val = data_dir / "sub_val.xyz"
    sub_test = data_dir / "sub_test.xyz"

    if args.force_resample or (not sub_train.exists()) or (not sub_val.exists()) or (not sub_test.exists()):
        logger.info("Selecting initial subset and splitting dataset")
        init_atoms = read((data_dir / "init_dataset.xyz").as_posix(), index=":")
        rng = np.random.default_rng(args.seed)
        subset_idx = rng.choice(len(init_atoms), size=args.subset_size, replace=False)
        subset = [init_atoms[i] for i in subset_idx]
        write((data_dir / "sub_dataset.xyz").as_posix(), subset, format="extxyz")

        split_idx = rng.permutation(len(subset))
        n_train = int(len(subset) * args.train_ratio)
        n_val = int(len(subset) * args.val_ratio)

        train_atoms = [subset[i] for i in split_idx[:n_train]]
        val_atoms = [subset[i] for i in split_idx[n_train:n_train + n_val]]
        test_atoms = [subset[i] for i in split_idx[n_train + n_val:]]

        write(sub_train.as_posix(), train_atoms, format="extxyz")
        write(sub_val.as_posix(), val_atoms, format="extxyz")
        write(sub_test.as_posix(), test_atoms, format="extxyz")

        logger.info("Init split: train=%d val=%d test=%d", len(train_atoms), len(val_atoms), len(test_atoms))
    else:
        logger.info("Found existing split: %s / %s / %s", sub_train, sub_val, sub_test)

    # We keep a "current train" file that grows with AL
    current_train = data_dir / "active_train.xyz"
    if not current_train.exists():
        shutil.copy2(sub_train, current_train)
        logger.info("Created %s from %s", current_train, sub_train)

    # ---------- Active-learning rounds ---------- #
    for rnd in range(args.n_rounds):
        logger.info("=" * 80)
        logger.info("ACTIVE LEARNING ROUND %d / %d", rnd + 1, args.n_rounds)
        logger.info("Current train set: %s", current_train)

        # 1) write configs (pointing to current_train, but val/test fixed)
        if len(args.comm_seed) < args.n_comm:
            raise ValueError("--comm_seed must provide at least n_comm seeds")
        cfg_paths = write_committee_configs(
            train_file=current_train,
            valid_file=sub_val,
            test_file=sub_test,
            n_comm=args.n_comm,
            comm_seed=args.comm_seed[:args.n_comm],
            foundation_model=Path(args.foundation_model),
            config_dir=Path("config"),
            committee_dir=Path("committee"),
            comm_epoch=args.comm_epoch,
            device = args.device,
        )

        # 2) train committees
        train_committees(cfg_paths)

        # 3) run MD with committee models
        model_paths = [Path(f"committee/model{i+1}/comm_v1_model{i + 1}.model") for i in range(args.n_comm)]
        for mp in model_paths:
            if not mp.exists():
                raise FileNotFoundError(f"Missing trained model: {mp}")

        if args.keep_all_rounds:
            round_tag = f"round_{rnd:02d}"
            traj_dir = Path("comm_traj") / round_tag
        else:
            traj_dir = Path("comm_traj")
        ensure_dir(traj_dir)

        md_traj = traj_dir / "ase_traj.xyz"
        run_ase_md(
            model_paths=model_paths,
            init_conf_path=Path(args.init_conf_path),
            md_temp=args.md_temp,
            md_steps=args.md_steps,
            traj_out=md_traj,
            interval=args.md_write_interval,
        )

        # 4) select uncertain frames
        traj_uncert = traj_dir / "ase_traj_with_uncertainty.xyz"
        uncertain_xyz = traj_dir / "uncertain_frames.xyz"
        n_unc = select_uncertain_frames(
            model_paths=model_paths,
            traj_in=md_traj,
            traj_out_with_uncert=traj_uncert,
            uncertain_out=uncertain_xyz,
            threshold=args.uncert_threshold,
            max_frames=args.uncert_max_frames,
        )
        if n_unc == 0:
            logger.warning("No uncertain frames found. Stopping early at round %d.", rnd + 1)
            break

        # 5) prepare DFT inputs
        dft_root = Path(args.dft_root)
        n_dft = prepare_dft_inputs(
            uncertain_xyz=uncertain_xyz,
            dft_root=dft_root,
            template_dir=Path(args.dft_template_dir),
            files_to_copy=list(args.dft_template_files),
            submitjobs_src=Path(args.submitjobs),
            outcar2xyz_src=Path(args.outcar2xyz),
        )
        if n_dft == 0:
            logger.warning("No DFT tasks prepared. Stopping early.")
            break

        # 6) submit jobs
        run_bash(script=dft_root / Path(args.submitjobs).name, cwd=dft_root, log_path=dft_root / "submitjobs.log")

        # 7) wait completion
        timeout_sec = int(args.dft_timeout_hr * 3600)
        done_dirs, failed_dirs = wait_for_dft(dft_root, poll_sec=args.dft_poll_sec, timeout_sec=timeout_sec)
        if failed_dirs:
            logger.warning("Some DFT jobs failed (%d). They will be skipped in merging.", len(failed_dirs))

        # 8) merge DFT results (prefer your outcar2xyz.sh)
        ensure_plain_outcars(dft_root)
        merged_from_sh = try_run_outcar2xyz(dft_root, read_dir='.')
        
# # If your outcar2xyz.sh marks non-converged OUTCARs, it writes this file.
# nonconv = dft_root / "non_converged_files.txt"
# if nonconv.exists():
#     try:
#         bad = [ln.strip() for ln in nonconv.read_text(encoding="utf-8").splitlines() if ln.strip()]
#         if bad:
#             logger.warning("outcar2xyz.sh found %d non-converged OUTCARs. See: %s", len(bad), nonconv)
#     except Exception as e:
#         logger.warning("Failed reading %s: %s", nonconv, e)

        merged_out = dft_root / args.merged_dft_name
        if merged_from_sh is not None and merged_from_sh.exists():
            # ensure extxyz suffix for MACE
            shutil.copy2(merged_from_sh, merged_out)
        else:
            merged_out = python_merge_from_outcar(dft_root, merged_out)

        if not merged_out.exists():
            raise RuntimeError("Merged DFT xyz was not produced")

        # 9) append to training set
        next_train = data_dir / f"active_train_round{rnd + 1:02d}.xyz"
        append_extxyz_concat(current_train, merged_out, next_train)
        logger.info("Updated training set -> %s (added %s)", next_train, merged_out)
        current_train = next_train

    logger.info("Workflow finished. Final train set: %s", current_train)


if __name__ == "__main__":
    main()
