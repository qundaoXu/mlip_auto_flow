"""The AIMD state points and their matched MLIP runs.

A "state point" pairs one AIMD reference segment with the MLIP run that mirrors
it: same starting configuration, same cell and density, same temperature
programme, same timestep.  Keeping the pairing in one place is what stops an
L3/L4/L5 comparison from quietly drifting into comparing two different physical
situations.

Since 2026-10-27 the active state points (:data:`STATES`) are the phase-1
canonical AIMD chains: 306-atom Ge68Sb68Te170 at 0.0309 at/A^3, three
independent replicas each.  MLIP seed ``k`` starts from replica ``k + 1``, so the
model side mirrors the reference side replica by replica, and the reference
side finally has the chain-to-chain spread that ``sigma_ref`` needs
(:mod:`mlipeval.ensemble`).

Six state points since 2026-11-02: the 300 K glass, the 950 / 1200 / 3000 K
liquids (L3 + L4 for the liquids) and two ramps for L5, the 1200 -> 300 K
quench and the 300 -> 1200 K heating.  The legacy single-chain state points
(0.02852 at/A^3) were retired the same day.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path

from . import datasets
from .io_vasp import Trajectory, concatenate
from .paths import DATA

#: MD 引擎。两个都是受支持的选项，物理上等价（温度、结构、RDF、最小间距、
#: 守恒量实测一致，见 ``scripts/experiments/lammps_md/README.md``）：
#:
#: * ``lammps`` —— 在生产引擎里跑，benchmark 测的是用户实际会遇到的东西；
#:   温度斜坡由 ``fix nvt temp T0 T1`` 原生支持。
#: * ``ase`` —— 单一代码路径，不依赖三套 LAMMPS 构建；MACE 快 1.45×、DPA 快 3×。
#:
#: 由 ``MLIPEVAL_MD_ENGINE`` 选择，与 ``MLIPEVAL_MODEL`` 同样是进程级开关。
#: **跨模型比较必须固定同一个引擎**——这是框架"唯一变量是模型"的前提。
MD_ENGINES = ("lammps", "ase")
DEFAULT_MD_ENGINE = "lammps"


def current_engine() -> str:
    """本进程使用的 MD 引擎（``MLIPEVAL_MD_ENGINE``）。"""
    engine = os.environ.get("MLIPEVAL_MD_ENGINE", "").strip() or DEFAULT_MD_ENGINE
    if engine not in MD_ENGINES:
        raise ValueError(
            f"未知 MD 引擎 {engine!r}；可选 {', '.join(MD_ENGINES)}"
        )
    return engine


@dataclass(frozen=True)
class StatePoint:
    name: str
    #: ``"aimd_306:<key>"`` / ``"aimd_216:<key>"`` for a legacy single chain, or
    #: ``"canon:<chain>:<stage>[+<stage>...]"`` for a phase-1 canonical chain,
    #: whose ``+``-joined stages are consecutive segments of one run.
    reference: str
    temperature_k: float
    temperature_end_k: float | None
    default_steps: int
    sample_every: int
    equilibration_fraction: float
    used_by: str
    note: str = ""
    #: Stride applied when the reference XDATCAR is read.  The canonical state
    #: points set it to ``sample_every``, so both sides are sampled at the same
    #: interval and the analysis strides below apply to both alike.
    reference_stride: int = 1
    #: Frame strides for the L3 battery: RDF/ADF/CN, and rings/ALTBC/S(q).
    l3_stride: int = 5
    ring_stride: int = 25
    #: How :func:`mlipeval.neighbors.first_minimum` picks the first RDF peak.
    #: The legacy state points keep ``"max"`` so their results stay reproducible.
    peak_rule: str = "max"

    @property
    def is_ramp(self) -> bool:
        return self.temperature_end_k is not None

    @property
    def is_canonical(self) -> bool:
        return self.reference.startswith("canon:")

    @property
    def n_replicas(self) -> int:
        return len(datasets.MELT_QUENCH_SEEDS) if self.is_canonical else 1

    @property
    def programme(self) -> tuple[tuple[float, float, float], ...]:
        """The temperature programme as ``(duration_fs, T0_K, T1_K)`` segments."""
        end = self.temperature_end_k if self.is_ramp else self.temperature_k
        return ((self.default_steps * 3.0, self.temperature_k, end),)

    def replica_for_seed(self, seed: int) -> int:
        """The AIMD replica an MLIP seed mirrors: seed ``k`` <-> replica ``k + 1``."""
        return seed % self.n_replicas + 1

    @property
    def md_path(self) -> Path:
        return self.md_path_for(0)

    def md_path_for(self, seed: int, engine: str | None = None) -> Path:
        """Saved MLIP run for one seed and one MD engine.

        The engine is **in the filename**, not only in the metadata, because the
        framework supports both (:mod:`mlipeval.md_lammps` and
        :mod:`mlipeval.md_runner`) and its central invariant is that a
        cross-model comparison varies only the model.  Two engines writing to one
        filename would let a re-run silently replace the other engine's
        trajectory, and the mix would only be visible by opening the file.
        """
        engine = engine or current_engine()
        suffix = "" if seed == 0 else f"_seed{seed}"
        return DATA / f"md_{self.name}__{engine}{suffix}.npz"

    def available_seeds(self, engine: str | None = None) -> list[int]:
        return [seed for seed in range(8) if self.md_path_for(seed, engine).exists()]


#: The active state points: phase-1 canonical chains, three replicas each.
#: Names carry a ``canon_`` *prefix* on purpose: result files are matched by
#: prefix (:func:`mlipeval.ranking._matching_stems`), and ``amorphous_300K_x``
#: would be swept up by the legacy ``amorphous_300K`` specs.
CANONICAL_STATES: dict[str, StatePoint] = {
    "canon_amorphous_300K": StatePoint(
        name="canon_amorphous_300K",
        reference="canon:H4_melt_quench:nvt_300K",
        temperature_k=300.0,
        temperature_end_k=None,
        default_steps=10_000,
        sample_every=15,
        equilibration_fraction=0.5,
        used_by="L3a/L3b",
        note="30 ps at 300 K after the 15 K/ps quench; the last 15 ps is production (TASKS.md)",
        reference_stride=15,
        peak_rule="first",
        l3_stride=1,
        ring_stride=6,
    ),
    "canon_liquid_1200K": StatePoint(
        name="canon_liquid_1200K",
        reference="canon:H4_melt_quench:nvt_1200K",
        temperature_k=1200.0,
        temperature_end_k=None,
        default_steps=10_000,
        sample_every=5,
        equilibration_fraction=0.25,
        used_by="L3a/L3b and L4",
        note="30 ps at 1200 K straight after the 3000 K melt; the first 7.5 ps is discarded",
        reference_stride=5,
        peak_rule="first",
        l3_stride=3,
        ring_stride=25,
    ),
    "canon_liquid_950K": StatePoint(
        name="canon_liquid_950K",
        reference="canon:H3_md950:equil_950K+prod_950K",
        temperature_k=950.0,
        temperature_end_k=None,
        default_steps=8_333,
        sample_every=5,
        # 3333 of 8333 steps: exactly the equilibration segment, so the
        # production window is the 15 ps H3 production run on both sides
        equilibration_fraction=0.4,
        used_by="L3a/L3b and L4",
        note="supercooled liquid: 10 ps equilibration + 15 ps production, velocities carried over",
        reference_stride=5,
        peak_rule="first",
        l3_stride=2,
        ring_stride=17,
    ),
    "canon_quench": StatePoint(
        name="canon_quench",
        reference="canon:H4_melt_quench:quench_1200_300K",
        temperature_k=1200.0,
        temperature_end_k=300.0,
        default_steps=20_000,
        sample_every=10,
        equilibration_fraction=0.0,
        used_by="L5",
        note="linear 1200 -> 300 K NVT ramp over 60 ps (15 K/ps)",
        reference_stride=10,
        peak_rule="first",
    ),
    # Added 2026-11-02.  Both start where their AIMD replica starts: the melt from
    # the random-insertion cell, the heating ramp from that replica's 300 K glass.
    "canon_liquid_3000K": StatePoint(
        name="canon_liquid_3000K",
        reference="canon:H4_melt_quench:melt_3000K",
        temperature_k=3000.0,
        temperature_end_k=None,
        default_steps=3_000,
        sample_every=5,
        # 2.25 of 9 ps discarded: the random-insertion start is forgotten within
        # ~1 ps at 3000 K, and the window left is 450 frames per replica
        equilibration_fraction=0.25,
        used_by="L3a/L3b and L4",
        note="9 ps at 3000 K from random insertion (seed 1: 638 archived + 2362 published steps)",
        reference_stride=5,
        peak_rule="first",
        l3_stride=1,
        ring_stride=8,
    ),
    "canon_heating": StatePoint(
        name="canon_heating",
        reference="canon:H3_heating:heating_300_1200K",
        temperature_k=300.0,
        temperature_end_k=1200.0,
        default_steps=8_333,
        sample_every=10,
        equilibration_fraction=0.0,
        used_by="L5",
        note="linear 300 -> 1200 K NVT ramp over 25 ps (~36 K/ps) from the H4 300 K glass",
        reference_stride=10,
        peak_rule="first",
    ),
}

#: Every addressable state point.  The pre-2026-10-27 legacy state points
#: (one 306-atom chain each at 0.02852 at/A^3) were retired on 2026-11-02 along
#: with every other non-phase-1 reference; their result files are archived
#: under ``results/legacy_retired_20261102/``.
STATES: dict[str, StatePoint] = dict(CANONICAL_STATES)

#: The state point whose frozen bond cutoffs the hot liquids and the ramps
#: inherit (see ``run_l3.py --cutoffs-from``).
CUTOFF_STATE = "canon_amorphous_300K"


def get_state(state: StatePoint | str) -> StatePoint:
    """Look a state point up by name."""
    if isinstance(state, StatePoint):
        return state
    if state in STATES:
        return STATES[state]
    raise KeyError(f"unknown state point '{state}'; known: {sorted(STATES)}")


def isothermal_states() -> list[str]:
    """Active state points that L3 can time-average (no ramps)."""
    return [n for n, s in CANONICAL_STATES.items() if not s.is_ramp]


def ramp_states() -> list[str]:
    """Active state points that L5 follows along their programme."""
    return [n for n, s in CANONICAL_STATES.items() if s.is_ramp]


def states_used_by(level: str) -> list[str]:
    """Active state points whose ``used_by`` names ``level`` (e.g. ``"L4"``)."""
    return [n for n, s in CANONICAL_STATES.items() if level in s.used_by]


def missing_seeds(state: StatePoint | str) -> list[int]:
    """MLIP seeds a state point still lacks for its full replica pairing.

    An analysis step in a merged PBS chain must not abort the chain because a
    state point was added after that chain's MD stage ran: it reports the gap
    and moves on, and the gap is closed by an MD top-up for that state point.
    """
    state = get_state(state)
    engine = current_engine()
    return [seed for seed in range(state.n_replicas) if not state.md_path_for(seed, engine).exists()]


def reference_trajectory(state: StatePoint | str, *, replica: int = 1, **kwargs) -> Trajectory:
    """Load the AIMD side of a state point: one replica, whole programme."""
    state = get_state(state)
    kind, key = state.reference.split(":", 1)
    if kind != "canon":
        raise KeyError(f"{state.name}: '{state.reference}' is not a phase-1 canonical reference")

    chain, stages = key.split(":", 1)
    kwargs.setdefault("stride", state.reference_stride)
    segments = []
    for stage in stages.split("+"):
        segments.append(datasets.canonical_aimd(datasets._stage_task(chain, replica, stage), **kwargs))
        if kwargs.get("max_frames") is not None:
            break  # a start-frame read never needs the later segments
    if len(segments) == 1:
        traj = segments[0]
    else:
        protocol = dataclasses.replace(
            segments[0].protocol,
            protocol_id=f"gst225_aimd_canonical_v1:{state.name}:r{replica}",
            provenance={**segments[0].protocol.provenance, "segments": [s.name for s in segments]},
        )
        traj = concatenate(segments, name=f"{state.name}_r{replica}", protocol=protocol)
    traj.name = f"aimd_{state.name}_r{replica}"
    return traj


def mlip_trajectory(state: StatePoint | str, *, seed: int = 0, require: bool = True) -> Trajectory | None:
    """Load one MLIP run of a state point from ``eval/data/md_<state>[_seedN].npz``.

    For a canonical state point seed ``k`` mirrors AIMD replica ``k + 1``
    (:meth:`StatePoint.replica_for_seed`).  For a legacy one, with a single AIMD
    chain, a second seed is the only way to tell how much of an observed
    difference is chain-to-chain variability rather than model error.

    If the saved file predates protocol persistence, a protocol is rebuilt from
    the state-point definition so that downstream metrics still record which run
    produced them.
    """
    from .md_runner import load_md
    from .schema import Protocol

    state = get_state(state)
    engine = current_engine()
    path = state.md_path_for(seed, engine)
    if not path.exists():
        if require:
            other = [e for e in MD_ENGINES if e != engine
                     and state.md_path_for(seed, e).exists()]
            hint = (
                f"\n（另一个引擎的同名轨迹存在：{', '.join(other)}。"
                "跨模型比较必须固定同一引擎，所以这里不会替你回退。"
                f"要用它就设 MLIPEVAL_MD_ENGINE={other[0]}。）" if other else ""
            )
            raise FileNotFoundError(
                f"{path} not found -- run "
                f"`python run_mlip_md.py --state {state.name} --engine {engine}"
                + (f" --seed {seed}" if seed else "")
                + "` first" + hint
            )
        return None
    result = load_md(path)
    # 文件名与文件内容都记了引擎；不一致说明有人改过文件名。
    # 空字符串 = 字段引入之前存的盘，来源未知，此时以文件名为准。
    recorded = getattr(result.settings, "engine", "") or ""
    if recorded and recorded != engine:
        raise ValueError(
            f"{path} 里记录的引擎是 {recorded!r}，但当前是 {engine!r}——"
            "文件名与内容对不上，不要继续读，先查是谁改的名。"
        )
    traj = result.trajectory
    if traj.protocol is None:
        traj.protocol = Protocol(
            protocol_id=result.settings.protocol_id or f"mlip_md_{state.name}",
            kind="mlip_md",
            description=f"MACE MD mirroring {state.reference}",
            ensemble="NVT",
            timestep_fs=result.settings.timestep_fs,
            n_steps=int(result.step[-1]) if result.step.size else None,
            temperature_k=(
                (state.temperature_k, state.temperature_end_k) if state.is_ramp else state.temperature_k
            ),
            postprocessing={"sample_every": result.settings.sample_every},
            provenance={"rebuilt_from": "state_points.STATES", "file": str(path), "seed": seed},
        )
    traj.name = f"mlip_{state.name}" + (f"_seed{seed}" if seed else "")
    return traj


def _production(state: StatePoint, traj: Trajectory) -> Trajectory:
    """The production window: the whole programme for a ramp, else the tail."""
    return traj if state.is_ramp else traj.last_fraction(1.0 - state.equilibration_fraction)


def reference_production(state: StatePoint | str, replica: int = 1, *, reference_stride: int | None = None) -> Trajectory:
    state = get_state(state)
    kwargs = {} if reference_stride is None else {"stride": reference_stride}
    return _production(state, reference_trajectory(state, replica=replica, **kwargs))


def model_production(state: StatePoint | str, seed: int = 0) -> Trajectory:
    state = get_state(state)
    return _production(state, mlip_trajectory(state, seed=seed))


def production_segments(
    state: StatePoint | str,
    *,
    reference_stride: int | None = None,
    seed: int | None = None,
    replica: int | None = None,
) -> dict[str, Trajectory]:
    """Reference and model production windows for one replica pairing.

    For an isothermal state point the trailing ``1 - equilibration_fraction`` of
    each trajectory is used on both sides; for a ramp the whole programme is
    kept, because the path itself is the observable.  With neither ``seed`` nor
    ``replica`` given this is replica 1 against seed 0, the pairing every
    single-chain driver used before replicas existed.
    """
    state = get_state(state)
    if replica is None:
        replica = state.replica_for_seed(seed or 0)
    if seed is None:
        seed = replica - 1
    return {
        "reference": reference_production(state, replica, reference_stride=reference_stride),
        "model": model_production(state, seed),
    }


def frozen_params(state: StatePoint | str, *, stride: int | None = None):
    """Freeze the L3 structure parameters from a state point's AIMD reference.

    One place for the stride and peak rule, so ``run_l3``, ``run_l3_uncertainty``
    and ``run_l5`` cannot freeze the same state point two different ways.
    Returns ``(StructureParams, reference partial RDF)``.
    """
    from .l3a_short_range import freeze_params_from_reference

    state = get_state(state)
    return freeze_params_from_reference(
        cutoff_reference(state), stride=stride or state.l3_stride, peak_rule=state.peak_rule
    )


def cutoff_reference(state: StatePoint | str) -> Trajectory:
    """The trajectory the L3 bond cutoffs are frozen from.

    For a canonical state point this pools the production windows of all
    replicas, so every model-vs-replica and replica-vs-replica comparison of
    :mod:`mlipeval.ensemble` shares one frozen parameter set rather than three
    slightly different ones.
    """
    state = get_state(state)
    windows = [reference_production(state, r) for r in range(1, state.n_replicas + 1)]
    if len(windows) == 1:
        return windows[0]
    return concatenate(windows, name=f"aimd_{state.name}_pooled")


#: The MD + L3-L5 analysis commands for one model, in order.  Shared by
#: ``submit_pbs.py``, ``submit_al_ladder.py`` and ``run_all.py`` so that the
#: three entry points cannot drift apart again.  The hot liquids and the quench
#: inherit the 300 K amorphous cutoffs (``CUTOFF_STATE``).
def md_commands() -> list[str]:
    return [f"run_mlip_md.py --state all --seed {seed}" for seed in range(len(datasets.MELT_QUENCH_SEEDS))]


def analysis_commands() -> list[str]:
    commands = []
    for name in isothermal_states():
        inherit = "" if name == CUTOFF_STATE else f" --cutoffs-from {CUTOFF_STATE}"
        commands.append(f"run_l3.py --state {name}{inherit}")
        commands.append(f"run_l3_uncertainty.py --state {name}{inherit}")
    commands += ["run_l4.py --state all --check-ramp", "run_l5.py --state all"]
    return commands
