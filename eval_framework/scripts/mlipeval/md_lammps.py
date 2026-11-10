"""LAMMPS 引擎的 MLIP MD，产出与 :mod:`mlipeval.md_runner` 完全相同的 ``MDResult``。

## 为什么换引擎

框架原来用 ASE 跑 MD。对照实验（``scripts/experiments/lammps_md/README.md``，作业 608082）
在同一状态点、同一模型、同一起始构型下测得：温度、结构、最小原子间距两个引擎一致，
但**扩展系统守恒量的漂移差 440 倍**——LAMMPS 的 ``fix nvt`` 是 −0.0012 meV/atom/ps，
ASE 的 ``NoseHooverChainNVT`` 是 +0.548。理由是物理的，不是速度的：
实测 LAMMPS 对 DPA-4-Pro 反而慢 3 倍。

另外两条：``fix nvt`` 比 ASE 那个需要手工改 ``_Q`` 的实现更接近 VASP 的 ``SMASS=0``；
温度斜坡（L5 melt–quench）在 LAMMPS 里是 ``temp T0 T1`` 原生支持。

## 三条路线

模型的 LAMMPS 接入方式不同，但**都以子进程方式启动**，所以这里只有一个执行路径：

===========  ==========================================  =========================
route        模型                                          启动
===========  ==========================================  =========================
``mliap``    MACE / NequIP（含四个专有势）                 ``python run_mliap.py``
``deepmd``   DPA-4                                        ``lmp``
``gnnp``     CHGNet / MatterSim（AdvanceSoft ML-GNNP）     AdvanceSoft ``lmp``（串行）
===========  ==========================================  =========================

``mliap`` 必须由 Python 主进程驱动（独立 ``lmp`` 会初始化第二个内嵌 CPython，
libtorch 被加载两次并以 "Only a single TORCH_LIBRARY can be used" 中止），
这正是 ``universal_MLIP/common/run_mliap.py`` 存在的原因。子进程模型让这条
特殊性被关在启动命令里，不外溢到框架。

## 与 ASE 版的行为差异（都是有意的）

* **速度初始化用的是 LAMMPS 的 RNG**，与 ASE 的 ``np.random.default_rng(seed)`` 不同。
  同一个 ``seed`` 在两个引擎下给出**不同的初速度实现**，所以两边的轨迹不可逐帧对照，
  只有统计量可比。种子仍然逐 run 固定，可复现。
* **最小原子间距在采样帧上事后算**，不是每步检查。ASE 版是每个采样点检查并当场中止；
  这里 LAMMPS 跑完再判，崩溃通过"步数没跑满/能量非有限"识别。判据本身没变。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms

from . import paths as _paths
from .io_vasp import Trajectory
from .md_runner import MDResult, MDSettings

#: pair_coeff 里写的元素顺序，必须与 start.data 的 type 编号一致。
SPECORDER = ["Ge", "Sb", "Te"]

#: Shared LAMMPS helper scripts (``lammps.common_dir`` in the site config).
COMMON = _paths.LAMMPS_COMMON
#: LAMMPS executable (``lammps.executable`` / ``$MLIPEVAL_LMP``).
ADVANCESOFT_LMP = _paths.LAMMPS_EXE

IN_MD = """\
# 由 mlipeval.md_lammps 生成——参数来自 MDSettings，勿手改
units           metal
atom_style      atomic
atom_modify     map yes
newton          on
boundary        p p p

read_data       {data}

include         {pair_in}

velocity        all create {t_start} {vseed} dist gaussian mom yes rot yes

timestep        {dt_ps}

{fix_line}

thermo_style    custom step time temp pe ke etotal econserve press vol
thermo          {sample_every}
thermo_modify   norm no

dump            traj all custom {sample_every} {dump} id type x y z xu yu zu
dump_modify     traj sort id

run             {n_steps}

print           "LAMMPS_MD_OK steps={n_steps}"
"""


@dataclass(frozen=True)
class LammpsRoute:
    """一个模型怎么在 LAMMPS 里跑起来。"""

    route: str          # "mliap" | "deepmd" | "gnnp"
    pair_in: Path       # 该模型的 pair.in
    serial_only: bool = False

    def command(self, in_md: Path) -> list[str]:
        if self.route == "mliap":
            # 必须 Python 主进程驱动，见模块 docstring
            return ["python", "-u", str(COMMON / "run_mliap.py"), str(in_md)]
        if self.route == "deepmd":
            return ["lmp", "-in", str(in_md)]
        if self.route == "gnnp":
            # pair_style gnnp 禁止 nprocs > 1（该 build 无 MPI）
            return [str(ADVANCESOFT_LMP), "-in", str(in_md)]
        raise ValueError(f"未知 LAMMPS 路线 {self.route!r}")


def route_for(entry) -> LammpsRoute:
    """从 :class:`mlipeval.models.ModelEntry` 取它的 LAMMPS 路线。"""
    if entry.lammps_route is None or entry.lammps_pair_in is None:
        raise ValueError(
            f"模型 {entry.key} 没有登记 LAMMPS 路线；"
            "md_lammps 需要 ModelEntry.lammps_route 与 .lammps_pair_in"
        )
    pair_in = Path(entry.lammps_pair_in)
    if not pair_in.is_file():
        raise FileNotFoundError(f"{entry.key} 的 pair.in 不存在: {pair_in}")
    return LammpsRoute(route=entry.lammps_route, pair_in=pair_in,
                       serial_only=entry.lammps_route == "gnnp")


def write_inputs(atoms: Atoms, settings: MDSettings, route: LammpsRoute, workdir: Path) -> Path:
    """落盘 start.data 与 in.md，返回 in.md 的路径（绝对路径，避免 cd 相关的坑）。"""
    from ase.io import write as ase_write

    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    syms = set(atoms.get_chemical_symbols())
    if not syms <= set(SPECORDER):
        raise ValueError(f"意外的元素 {sorted(syms - set(SPECORDER))}；specorder 是 {SPECORDER}")

    data = workdir / "start.data"
    ase_write(data, atoms, format="lammps-data", specorder=SPECORDER, masses=True)

    t_end = settings.temperature_end_k if settings.temperature_end_k is not None else settings.temperature_k
    if settings.ensemble == "nve":
        fix_line = "fix             md all nve"
    else:
        # tdamp 在 metal units 下是 ps
        fix_line = (
            f"fix             md all nvt temp {settings.temperature_k:.4f} "
            f"{t_end:.4f} {settings.effective_tdamp_fs / 1000.0:.6f}"
        )

    in_md = workdir / "in.md"
    in_md.write_text(
        IN_MD.format(
            data=data,
            pair_in=route.pair_in,
            t_start=f"{settings.temperature_k:.4f}",
            # LAMMPS 的 velocity seed 必须是正整数；ASE 那边 seed 0 是合法的
            vseed=20260909 + int(settings.seed),
            dt_ps=settings.timestep_fs / 1000.0,
            fix_line=fix_line,
            sample_every=settings.sample_every,
            n_steps=settings.n_steps,
            dump=workdir / "traj.dump",
        ),
        encoding="utf-8",
    )
    return in_md


#: thermo_style 里写的关键字与 LAMMPS **打印出来的表头**不是一回事：
#: `pe ke etotal` 打出来是 `PotEng KinEng TotEng`。按打印名归一到关键字名。
_THERMO_ALIASES = {
    "poteng": "pe",
    "kineng": "ke",
    "toteng": "etotal",
    "volume": "vol",
    "e_pair": "epair",
}


def parse_thermo(log_text: str) -> dict[str, np.ndarray]:
    """抓 thermo 表。LAMMPS 可能打印多段（minimize/run），取最后一段最长的。"""
    header: list[str] | None = None
    rows: list[list[float]] = []
    best: tuple[list[str], list[list[float]]] | None = None
    for line in log_text.splitlines():
        s = line.strip()
        if s.startswith("Step "):
            if header and rows and (best is None or len(rows) > len(best[1])):
                best = (header, rows)
            header, rows = s.split(), []
            continue
        if header is None:
            continue
        parts = s.split()
        if len(parts) == len(header):
            try:
                rows.append([float(x) for x in parts])
                continue
            except ValueError:
                pass
    if header and rows and (best is None or len(rows) > len(best[1])):
        best = (header, rows)
    if not best:
        raise RuntimeError("没有从 LAMMPS 日志里解析到 thermo 表")
    head, data = best
    arr = np.array(data)
    cols: dict[str, np.ndarray] = {}
    for i, name in enumerate(head):
        low = name.lower()
        cols[low] = arr[:, i]
        if low in _THERMO_ALIASES:
            cols[_THERMO_ALIASES[low]] = arr[:, i]
    missing = {"step", "temp", "pe", "ke", "econserve"} - cols.keys()
    if missing:
        raise RuntimeError(
            f"thermo 表缺列 {sorted(missing)}；实际表头是 {head}。"
            "in.md 的 thermo_style 与这里的解析必须一起改。"
        )
    return cols


def _min_distances(traj: Trajectory) -> np.ndarray:
    from ase.neighborlist import neighbor_list

    out = []
    for at in traj.iter_atoms():
        d = neighbor_list("d", at, 3.5)
        out.append(float(d.min()) if len(d) else float("nan"))
    return np.array(out)


def run_md_lammps(
    atoms: Atoms,
    model,
    settings: MDSettings,
    *,
    entry,
    workdir: Path,
    keep_workdir: bool = False,
    progress=None,
) -> MDResult:
    """跑一段 LAMMPS MD，返回与 ASE 版同构的 :class:`MDResult`。

    崩溃（LAMMPS 非零退出、步数没跑满、能量非有限）通过 ``crashed_at`` 报告而不是抛异常：
    不稳定的模型是一个**发现**，不是一个错误——与 ASE 版的语义一致。
    """
    from ase.io import read as ase_read

    route = route_for(entry)
    workdir = Path(workdir).resolve()
    in_md = write_inputs(atoms, settings, route, workdir)
    log = workdir / "run.log"
    dump = workdir / "traj.dump"
    if dump.exists():
        dump.unlink()

    cmd = route.command(in_md)
    if route.route != "gnnp" and shutil.which(cmd[0]) is None and not Path(cmd[0]).exists():
        raise FileNotFoundError(f"找不到 LAMMPS 启动程序 {cmd[0]}（route={route.route}）")

    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", str(os.environ.get("NCPUS", "12")))

    t0 = time.time()
    with open(log, "w") as fh:
        proc = subprocess.run(cmd, cwd=workdir, stdout=fh, stderr=subprocess.STDOUT, env=env)
    wall = time.time() - t0

    log_text = log.read_text(errors="replace")
    crashed_at: int | None = None
    crash_reason: str | None = None
    if proc.returncode != 0 and "LAMMPS_MD_OK" not in log_text:
        crash_reason = f"LAMMPS exit {proc.returncode}"
    err = re.search(r"^ERROR.*$", log_text, re.M)
    if err and crash_reason is None:
        crash_reason = err.group(0)[:200]

    cols = parse_thermo(log_text)
    frames = ase_read(str(dump), index=":", format="lammps-dump-text", specorder=SPECORDER)
    n = min(len(frames), len(cols["step"]))
    frames = frames[:n]
    positions = np.array([f.get_positions() for f in frames])
    cells = np.array([f.cell.array for f in frames])

    step = cols["step"][:n].astype(int)
    time_fs = step * settings.timestep_fs
    epot = cols["pe"][:n]
    ekin = cols["ke"][:n]
    econs = cols["econserve"][:n]
    temp = cols["temp"][:n]

    traj = Trajectory(
        symbols=list(frames[0].get_chemical_symbols()),
        positions=positions,
        cells=cells,
        name=settings.protocol_id,
        protocol=settings.to_protocol(model),
        timestep_fs=settings.timestep_fs,
        sample_stride=settings.sample_every,
        energies=epot,
        temperatures=temp,
        source=f"{model.spec.name} MD (LAMMPS/{route.route})",
    )
    dmin = _min_distances(traj)
    com0 = positions[0].mean(axis=0)
    com = np.linalg.norm(positions.mean(axis=1) - com0, axis=1)

    # dump 每 sample_every 步一帧，n_steps 不整除时最后一帧落在最后一个整倍数上
    last_sampled = settings.n_steps - settings.n_steps % settings.sample_every
    if crash_reason is None and int(step[-1]) < last_sampled:
        crash_reason = f"只跑到 {int(step[-1])}/{settings.n_steps} 步"
    if crash_reason is None and not np.isfinite(epot).all():
        crash_reason = "non-finite energy"
    bad = np.isfinite(dmin) & (dmin < settings.min_distance_floor)
    if crash_reason is None and bad.any():
        k = int(np.argmax(bad))
        crash_reason = f"unphysical minimum distance {dmin[k]:.3f} A"
        crashed_at = int(step[k])
    if crash_reason is not None and crashed_at is None:
        crashed_at = int(step[-1])

    if not keep_workdir and crash_reason is None:
        # 轨迹已经进了 MDResult，dump 是最大的中间产物
        dump.unlink(missing_ok=True)

    if progress:
        progress(int(step[-1]), settings.n_steps)

    return MDResult(
        trajectory=traj,
        settings=settings,
        step=step,
        time_fs=time_fs,
        potential_energy=epot,
        kinetic_energy=ekin,
        total_energy=econs,
        temperature=temp,
        min_distance=dmin,
        com_drift=com,
        wall_time_s=wall,
        crashed_at=crashed_at,
        crash_reason=crash_reason,
    )
