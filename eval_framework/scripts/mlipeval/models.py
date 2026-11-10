"""Registry of the universal MLIPs under test.

Phase 1 of this framework was written around a single model (MACE-MPA-0).  The
five models compared here do not share a Python environment -- ``nequip`` pins
e3nn 0.5.x while ``mace-torch`` pins 0.4.4, and ``deepmd-kit`` pins torch 2.11 --
so each one carries the conda prefix its calculator needs.  A driver therefore
runs *inside* the model's environment, selected by ``MLIPEVAL_MODEL``:

    MLIPEVAL_MODEL=nequip_oam_l python run_l0.py

:mod:`mlipeval.paths` reads the same variable and redirects ``results/`` and
``data/`` into ``models/<key>/``, so five runs of the identical code never write
over each other.  ``data/cache`` stays shared: its keys already contain the
checkpoint hash.

The checkpoint recorded here is what gets sha256'd into every ``ModelSpec``; for
DPA-4 that is deliberately the *training checkpoint*, not the frozen artefact,
because the frozen file is device- and compiler-specific while the checkpoint is
the thing the Matbench Discovery entry names.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from . import config
from .paths import AL_LADDER_ROOT, GST_AGING_ROOT, MODEL_ROOT, SPECIALISED_ROOT

#: Conda environments: ``environment.conda_envs`` in the site config
#: (``$MLIPEVAL_CONDA_ENVS``), default ``<conda_root>/envs``.
ENVS = config.conda_envs()

#: Selected by ``MLIPEVAL_MODEL``; the historical default keeps phase-1 behaviour.
DEFAULT_MODEL = "mace_mpa0"


@dataclass(frozen=True)
class ModelEntry:
    """Everything needed to build one model and to name its results."""

    key: str
    label: str
    backend: str  # "mace" | "nequip" | "deepmd" | "mattersim" | "chgnet"
    checkpoint: Path  # hashed into ModelSpec: the model's identity
    conda_env: Path
    native_protocol: str
    dtype: str
    load_path: Path | None = None  # what the calculator actually opens, if different
    #: How this model runs MD in LAMMPS: "mliap" | "deepmd" | "gnnp", plus the
    #: ``pair.in`` holding its pair_style/pair_coeff.  L3/L4/L5 MD goes through
    #: LAMMPS (:mod:`mlipeval.md_lammps`) because its Nose-Hoover conserves the
    #: extended-system energy ~440x better than ASE's -- measured, see
    #: ``scripts/experiments/lammps_md/README.md``.  Static single points stay on
    #: the ASE calculator: L0/L1/L2 need per-frame energies and forces on
    #: arbitrary cells, which is what a calculator is for.
    lammps_route: str | None = None
    lammps_pair_in: Path | None = None
    notes: str = ""

    @property
    def loadable(self) -> Path:
        return self.load_path or self.checkpoint


REGISTRY: dict[str, ModelEntry] = {
    "mace_mpa0": ModelEntry(
        key="mace_mpa0",
        label="MACE-MPA-0 (medium)",
        backend="mace",
        # 2026-09-09: 用户把这个文件从 universal_MLIP/ 顶层移进了 mace_mpa0/，
        # 与其余四个通用势的一模型一目录布局对齐。sha256 未变
        # （75428afe3a…，与历史结果和 data/cache 的键一致），所以只是路径变了。
        checkpoint=MODEL_ROOT / "mace_mpa0/mace-mpa-0-medium.model",
        conda_env=config.backend_env("mace"),
        lammps_route="mliap",
        lammps_pair_in=MODEL_ROOT / "mace_mpa0/pair.in",
        native_protocol="PBE+U (MPtrj + sAlex), no dispersion",
        dtype="float64",
        notes="phase-1 baseline; the only model this framework was originally written for",
    ),
    "nequip_oam_l": ModelEntry(
        key="nequip_oam_l",
        label="NequIP-OAM-L 0.1",
        backend="nequip",
        checkpoint=MODEL_ROOT / "nequip_oam_l/NequIP-OAM-L-0.1.nequip.zip",
        conda_env=config.backend_env("nequip"),
        lammps_route="mliap",
        lammps_pair_in=MODEL_ROOT / "nequip_oam_l/pair.in",
        native_protocol="PBE / PBE+U (OMat24 + MPtrj + sAlex), no dispersion",
        dtype="float64",
        notes="loaded eagerly from the nequip package archive; ZBL short-range core",
    ),
    "dpa4_pro": ModelEntry(
        key="dpa4_pro",
        label="DPA-4.0.1-Pro-MPtrj",
        backend="deepmd",
        checkpoint=MODEL_ROOT / "dpa4_pro/dpa4.0.1-pro-mptrj.pt",
        conda_env=config.backend_env("deepmd"),
        lammps_route="deepmd",
        lammps_pair_in=MODEL_ROOT / "dpa4_pro/pair.in",
        native_protocol="PBE+U (MPtrj), no dispersion",
        dtype="float64",
        notes=(
            "22.8M parameters, evaluated eagerly from the training checkpoint. "
            "The AOTInductor archive that `dp --pt freeze` produces is the right "
            "artefact for LAMMPS but not for this framework: its baked-in dynamic "
            "shape guards abort on small systems (a two-atom dimer, which L1's PEC "
            "scan needs), surfacing as a spurious out-of-memory error."
        ),
    ),
    "mattersim_5m": ModelEntry(
        key="mattersim_5m",
        label="MatterSim v1 5M",
        backend="mattersim",
        checkpoint=MODEL_ROOT / "mattersim/mattersim-v1.0.0-5M.pth",
        conda_env=config.backend_env("mattersim"),
        lammps_route="gnnp",
        lammps_pair_in=MODEL_ROOT / "mattersim/pair.in",
        native_protocol="PBE (MPtrj + in-house finite-T/P set), no dispersion",
        dtype="float32",
        notes="M3GNet-style; trained on finite-temperature and finite-pressure structures",
    ),
    "chgnet_030": ModelEntry(
        key="chgnet_030",
        label="CHGNet 0.3.0",
        backend="chgnet",
        checkpoint=MODEL_ROOT / "chgnet/chgnet_0.3.0_e29f68s314m37.pth.tar",
        conda_env=config.backend_env("chgnet"),
        lammps_route="gnnp",
        lammps_pair_in=MODEL_ROOT / "chgnet/pair.in",
        native_protocol="PBE / PBE+U (MPtrj), no dispersion, magmom head",
        dtype="float32",
        notes="0.4M parameters; energies are MP-corrected, so its zero point differs strongly",
    ),
    # ---- specialised potentials: not universal MLIPs, see SPECIALISED -----
    "mace_spec_gst": ModelEntry(
        key="mace_spec_gst",
        label="MACE-spec-GST (GST-ACE-24)",
        backend="mace",
        checkpoint=(
            SPECIALISED_ROOT
            / "mace_spec_gst/model/mace_spec_gst.model"
        ),
        conda_env=config.backend_env("mace"),
        lammps_route="mliap",
        lammps_pair_in=SPECIALISED_ROOT / "mace_spec_gst/pair.in",
        native_protocol="PBE (GST-ACE-24, protocol GSTACE24-VASP-PBE), no dispersion",
        dtype="float64",
        notes=(
            "MACE trained on the public GST-ACE-24 set (4171 training frames), "
            "architecture aligned item-by-item to MACE-MPA-0 so that training data is "
            "the only variable. Best-validation checkpoint, epoch 155. "
            "NOT a universal potential and NOT part of ORDER: it sees only Ge/Sb/Te, "
            "and two of this framework's reference sets are inside its training data "
            "(see mlipeval.leakage.GST_ACE24_LEAKAGE)."
        ),
    ),
    "mace_spec_gst_ne": ModelEntry(
        key="mace_spec_gst_ne",
        label="MACE-spec-GST-NE (processed_NE)",
        backend="mace",
        checkpoint=(
            SPECIALISED_ROOT
            / "mace_spec_gst_ne/model/mace_spec_gst_ne.model"
        ),
        conda_env=config.backend_env("mace"),
        lammps_route="mliap",
        lammps_pair_in=SPECIALISED_ROOT / "mace_spec_gst_ne/pair.in",
        native_protocol="PBE (processed_NE, protocol PROCNE-VASP-PBE), no dispersion",
        dtype="float64",
        notes=(
            "Same architecture and hyperparameters as mace_spec_gst -- only the "
            "training data differs (processed_NE, 2276 training frames). "
            "Best-validation checkpoint, epoch 205; the first run where the patched "
            "early stopping actually fired (stopped at 405 of a 1500 cap). "
            "Its training set is near-equilibrium only (no rss/random/dimer frames), "
            "so unlike mace_spec_gst its L1 PEC is NOT contaminated; the one leak it "
            "shares is the L2 hexagonal reference "
            "(see mlipeval.leakage.PROCESSED_NE_LEAKAGE). "
            "NOT a universal potential and NOT part of ORDER."
        ),
    ),
    "nequip_spec_gst": ModelEntry(
        key="nequip_spec_gst",
        label="NequIP-spec-GST (GST-ACE-24)",
        backend="nequip",
        checkpoint=(
            SPECIALISED_ROOT
            / "nequip/run_gst_ace/nequip_spec_gst.nequip.zip"
        ),
        conda_env=config.backend_env("nequip"),
        lammps_route="mliap",
        lammps_pair_in=SPECIALISED_ROOT / "nequip/mliap/pair.in",
        native_protocol="PBE (GST-ACE-24, protocol GSTACE24-VASP-PBE), no dispersion",
        dtype="float64",
        notes=(
            "NequIP trained on GST-ACE-24 (the same 4171 training frames as "
            "mace_spec_gst), architecture read item-by-item off NequIP-OAM-L so that "
            "training data is the only variable against that universal potential -- "
            "the NequIP-side counterpart of mace_spec_gst vs mace_mpa0. "
            "9.6M parameters, 6 conv layers, r_max 6.0, ZBL, EMA weights of the "
            "best-validation checkpoint (epoch 551 of 752; early stopping fired). "
            "`dtype` is the framework-level IO dtype, matching nequip_oam_l so the two "
            "NequIP entries stay comparable; both models are internally model_dtype "
            "float32, which is OAM-L's own setting (the MACE specialised potentials are "
            "float64 because MACE-MPA-0 is). "
            "Shares mace_spec_gst's training set exactly, so it carries the identical "
            "leakage findings (mlipeval.leakage.GST_ACE24_LEAKAGE). "
            "NOT a universal potential and NOT part of ORDER."
        ),
    ),
    "mace_gst_aging": ModelEntry(
        key="mace_gst_aging",
        label="MACE-GST225-Aging",
        backend="mace",
        checkpoint=(
            (GST_AGING_ROOT / "01-model")
            / "GST225_Aging_Model.model"
        ),
        conda_env=config.backend_env("mace"),
        lammps_route="mliap",
        lammps_pair_in=SPECIALISED_ROOT / "mace_gst_aging/pair.in",
        native_protocol="PBE (GST225-aging set), no dispersion",
        dtype="float64",
        notes=(
            "Third-party GST225 aging potential, NOT architecture-aligned to MACE-MPA-0: "
            "no ZBL pair repulsion, no Agnesi distance transform, and plain "
            "RealAgnostic{,Residual}InteractionBlock rather than the Density variants. "
            "0.75M parameters, Ge/Sb/Te only, r_max 6.0, float64. "
            "Trained with --E0s=average, so atomic_energies are fitted bulk references "
            "(-4.199/-3.855/-3.036) and its energy zero differs from every other "
            "specialised potential here. "
            "IMPORTANT: trained with --stress_weight=0.0 --compute_stress=False, so it "
            "never saw a stress label; stress-derived metrics (L0 stress, L2 EOS/B0) are "
            "pure extrapolation and must be read as such. "
            "The path the user supplied was the LAMMPS mliap artefact "
            "(GST225_Aging_Model.model-mliap_lammps.pt); this framework needs the native "
            "MACE .model, which sits beside it. "
            "Leakage: only the L2 hexagonal reference (see leakage.GST_AGING_LEAKAGE). "
            "NOT a universal potential and NOT part of ORDER."
        ),
    ),

# ---- the GST225 active-learning ladder -------------------------------------
# One model per point of the replayed AL ladder (al_replay_GST225, bwlab), all
# seed 1 and all the stage-two/SWA checkpoint. They share ONE architecture, ONE
# set of hyperparameters and ONE optimiser schedule; the training set is the
# only thing that changes, which is what makes the sequence readable as the
# effect of active learning rather than of retuning.
#
# They are siblings of `mace_gst_aging`: same architecture family, and al_prod's
# training set is the deployed model's own 4577-frame set after de-duplication.
# `mace_gst_aging` is therefore the natural reference point for the top of the
# ladder, not a separate lineage.
#
# Trained in float32 for speed; `dtype` here is the framework IO dtype and is
# float64 to match every other MACE entry, so the calculator upcasts. That was
# the same choice the bwlab held-out evaluation made, so the two sets of numbers
# are directly comparable.
#
# NOT universal potentials and NOT part of ORDER.

    "al_gen0": ModelEntry(
        key="al_gen0",
        label="AL gen0 (422 frames)",
        backend="mace",
        checkpoint=AL_LADDER_ROOT / "model/al_gen0.model",
        conda_env=config.backend_env("mace"),
        lammps_route="mliap",
        lammps_pair_in=AL_LADDER_ROOT / "pair/al_gen0.in",
        native_protocol="PBE (GST225-aging lineage), no dispersion",
        dtype="float64",
        notes=(
            "AL ladder point, training set gen0.xyz. Same architecture and "
            "hyperparameters as every other ladder point; only the training data "
            "differs. Trained float32, evaluated float64. Like mace_gst_aging it "
            "never saw a stress label, so L0 stress and L2 EOS/B0 are extrapolation."
        ),
    ),
    "al_gen3": ModelEntry(
        key="al_gen3",
        label="AL gen3 (780 frames)",
        backend="mace",
        checkpoint=AL_LADDER_ROOT / "model/al_gen3.model",
        conda_env=config.backend_env("mace"),
        lammps_route="mliap",
        lammps_pair_in=AL_LADDER_ROOT / "pair/al_gen3.in",
        native_protocol="PBE (GST225-aging lineage), no dispersion",
        dtype="float64",
        notes=(
            "AL ladder point, training set gen3.xyz. Same architecture and "
            "hyperparameters as every other ladder point; only the training data "
            "differs. Trained float32, evaluated float64. Like mace_gst_aging it "
            "never saw a stress label, so L0 stress and L2 EOS/B0 are extrapolation."
        ),
    ),
    "al_gen4": ModelEntry(
        key="al_gen4",
        label="AL gen4 (1060, off-path)",
        backend="mace",
        checkpoint=AL_LADDER_ROOT / "model/al_gen4.model",
        conda_env=config.backend_env("mace"),
        lammps_route="mliap",
        lammps_pair_in=AL_LADDER_ROOT / "pair/al_gen4.in",
        native_protocol="PBE (GST225-aging lineage), no dispersion",
        dtype="float64",
        notes=(
            "AL ladder point, training set gen4.xyz. Same architecture and "
            "hyperparameters as every other ladder point; only the training data "
            "differs. Trained float32, evaluated float64. Like mace_gst_aging it "
            "never saw a stress label, so L0 stress and L2 EOS/B0 are extrapolation."
        ),
    ),
    "al_gen6": ModelEntry(
        key="al_gen6",
        label="AL gen6 (1013 frames)",
        backend="mace",
        checkpoint=AL_LADDER_ROOT / "model/al_gen6.model",
        conda_env=config.backend_env("mace"),
        lammps_route="mliap",
        lammps_pair_in=AL_LADDER_ROOT / "pair/al_gen6.in",
        native_protocol="PBE (GST225-aging lineage), no dispersion",
        dtype="float64",
        notes=(
            "AL ladder point, training set gen6.xyz. Same architecture and "
            "hyperparameters as every other ladder point; only the training data "
            "differs. Trained float32, evaluated float64. Like mace_gst_aging it "
            "never saw a stress label, so L0 stress and L2 EOS/B0 are extrapolation."
        ),
    ),
    "al_gen7": ModelEntry(
        key="al_gen7",
        label="AL gen7 (1161 frames)",
        backend="mace",
        checkpoint=AL_LADDER_ROOT / "model/al_gen7.model",
        conda_env=config.backend_env("mace"),
        lammps_route="mliap",
        lammps_pair_in=AL_LADDER_ROOT / "pair/al_gen7.in",
        native_protocol="PBE (GST225-aging lineage), no dispersion",
        dtype="float64",
        notes=(
            "AL ladder point, training set gen7.xyz. Same architecture and "
            "hyperparameters as every other ladder point; only the training data "
            "differs. Trained float32, evaluated float64. Like mace_gst_aging it "
            "never saw a stress label, so L0 stress and L2 EOS/B0 are extrapolation."
        ),
    ),
    "al_gen8": ModelEntry(
        key="al_gen8",
        label="AL gen8 (701, pruning)",
        backend="mace",
        checkpoint=AL_LADDER_ROOT / "model/al_gen8.model",
        conda_env=config.backend_env("mace"),
        lammps_route="mliap",
        lammps_pair_in=AL_LADDER_ROOT / "pair/al_gen8.in",
        native_protocol="PBE (GST225-aging lineage), no dispersion",
        dtype="float64",
        notes=(
            "AL ladder point, training set gen8.xyz. Same architecture and "
            "hyperparameters as every other ladder point; only the training data "
            "differs. Trained float32, evaluated float64. Like mace_gst_aging it "
            "never saw a stress label, so L0 stress and L2 EOS/B0 are extrapolation."
        ),
    ),
    "al_gen9": ModelEntry(
        key="al_gen9",
        label="AL gen9 (1020 frames)",
        backend="mace",
        checkpoint=AL_LADDER_ROOT / "model/al_gen9.model",
        conda_env=config.backend_env("mace"),
        lammps_route="mliap",
        lammps_pair_in=AL_LADDER_ROOT / "pair/al_gen9.in",
        native_protocol="PBE (GST225-aging lineage), no dispersion",
        dtype="float64",
        notes=(
            "AL ladder point, training set gen9.xyz. Same architecture and "
            "hyperparameters as every other ladder point; only the training data "
            "differs. Trained float32, evaluated float64. Like mace_gst_aging it "
            "never saw a stress label, so L0 stress and L2 EOS/B0 are extrapolation."
        ),
    ),
    "al_prod": ModelEntry(
        key="al_prod",
        label="AL prod_train (3864)",
        backend="mace",
        checkpoint=AL_LADDER_ROOT / "model/al_prod.model",
        conda_env=config.backend_env("mace"),
        lammps_route="mliap",
        lammps_pair_in=AL_LADDER_ROOT / "pair/al_prod.in",
        native_protocol="PBE (GST225-aging lineage), no dispersion",
        dtype="float64",
        notes=(
            "AL ladder point, training set prod_train.xyz. Same architecture and "
            "hyperparameters as every other ladder point; only the training data "
            "differs. Trained float32, evaluated float64. Like mace_gst_aging it "
            "never saw a stress label, so L0 stress and L2 EOS/B0 are extrapolation."
        ),
    ),
}

#: Display order used by every table and figure in the ranking report.
#: Deliberately the five *universal* potentials only -- adding a GST-specialised
#: model here would put a potential that trained on this system's own chemistry
#: (and on two of the reference sets, see :mod:`mlipeval.leakage`) into the same
#: column as models that never saw it, and the headline number would be read as
#: if it meant the same thing.  Pairwise comparisons are run explicitly:
#:
#:     python run_ranking.py --models mace_spec_gst,mace_mpa0
ORDER = ["mace_mpa0", "nequip_oam_l", "dpa4_pro", "mattersim_5m", "chgnet_030"]

#: Registered but outside :data:`ORDER`.
SPECIALISED = ["mace_spec_gst", "mace_spec_gst_ne", "nequip_spec_gst", "mace_gst_aging"]

#: The active-learning ladder, in the order the generations were produced.
#: al_gen4 sits outside the curated path and is listed last on purpose.
AL_LADDER = ["al_gen0", "al_gen3", "al_gen6", "al_gen7", "al_gen8", "al_gen9",
             "al_prod", "al_gen4"]
SPECIALISED = SPECIALISED + AL_LADDER


# ---- models registered in the site config ----------------------------------
def _resolve(value, base: Path | None = None) -> Path | None:
    if value in (None, ""):
        return None
    p = Path(os.path.expandvars(str(value))).expanduser()
    return p if p.is_absolute() else ((base or config.workspace()) / p)


def _register_config_models() -> None:
    """Add (or override) registry entries from the ``models:`` section.

    Each entry needs ``label``, ``backend`` and ``checkpoint``; ``group``
    (``universal`` | ``specialised``) decides which list the key joins.
    Relative paths resolve against ``paths.workspace``.
    """
    for key, spec in (config.get("models", {}) or {}).items():
        spec = dict(spec or {})
        missing = [f for f in ("label", "backend", "checkpoint") if not spec.get(f)]
        if missing:
            raise ValueError(f"site config: model {key!r} is missing {', '.join(missing)}")
        REGISTRY[key] = ModelEntry(
            key=key,
            label=spec["label"],
            backend=spec["backend"],
            checkpoint=_resolve(spec["checkpoint"]),
            load_path=_resolve(spec.get("load_path")),
            conda_env=(config.conda_env(spec["conda_env"]) if spec.get("conda_env")
                       else config.backend_env(spec["backend"])),
            native_protocol=spec.get("native_protocol", "unspecified"),
            dtype=spec.get("dtype", "float64"),
            lammps_route=spec.get("lammps_route"),
            lammps_pair_in=_resolve(spec.get("lammps_pair_in")),
            notes=spec.get("notes", ""),
        )
        target = ORDER if spec.get("group", "specialised") == "universal" else SPECIALISED
        if key not in ORDER and key not in SPECIALISED:
            target.append(key)


_register_config_models()


def current_key() -> str:
    """Model selected for this process (``MLIPEVAL_MODEL``)."""
    return os.environ.get("MLIPEVAL_MODEL", "").strip() or DEFAULT_MODEL


def current() -> ModelEntry:
    return get(current_key())


def get(key: str) -> ModelEntry:
    try:
        return REGISTRY[key]
    except KeyError:
        raise KeyError(
            f"unknown model {key!r}; universal: {', '.join(ORDER)}; "
            f"specialised: {', '.join(SPECIALISED)}"
        ) from None


def all_entries() -> list[ModelEntry]:
    return [REGISTRY[k] for k in ORDER]
