"""Loaders that bind each reference dataset to its ``protocol_id``.

Every function here returns data that already knows which protocol produced it
and what role it may play.  Notebooks and drivers should go through this module
rather than reading files directly, so that a trajectory can never arrive at an
analysis without its protocol attached.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from ase import Atoms

from . import protocols
from .io_labeled import LabelledSet, decode_gst_config_types, read_labelled_extxyz
from .io_vasp import (
    Trajectory,
    concatenate,
    outcar_is_complete,
    read_incar,
    read_outcar_final,
    read_poscar,
    read_vasp_md_dir,
    read_xdatcar,
    task_output_dir,
)
from .paths import (
    AIMD_216,
    AIMD_306,
    CUBIC_SEEDS,
    DIAGNOSTIC_LABELS,
    DIAGNOSTIC_LEAKAGE,
    H5_LABELS,
    H5_LEAKAGE,
    LABELLED_615,
    MP_STRUCTURES,
    PHASE1,
)
from .schema import DataRole

#: The phase-1 melt-quench temperature programme, ``(duration_fs, T0_K, T1_K)``.
#: 306-atom Ge68Sb68Te170 at 0.0309 at/A^3, NVT throughout, POTIM 3.0 fs
#: (``phase1_vasp_inputs_20260828/TASKS.md`` section 2).
MELT_QUENCH_SEGMENTS = (
    (9_000.0, 3000.0, 3000.0),    # melt
    (30_000.0, 1200.0, 1200.0),   # equilibrate
    (60_000.0, 1200.0, 300.0),    # quench, 15 K/ps
    (30_000.0, 300.0, 300.0),     # 300 K, second half is production
)

#: Project standard amorphous number density (TASKS.md section 2).
STANDARD_NUMBER_DENSITY = 0.0309
STANDARD_N_ATOMS = 306
STANDARD_VOLUME_A3 = STANDARD_N_ATOMS / STANDARD_NUMBER_DENSITY


# --------------------------------------------------------------------------
# labelled single points
# --------------------------------------------------------------------------
def labelled_615(*, role: DataRole = DataRole.DIAGNOSTIC, decode_tags: bool = True) -> LabelledSet:
    """The 615-frame labelled set with its recovered protocol.

    Role is ``diagnostic`` by default.  It must not be promoted to
    ``protected_audit`` without a leakage screen against the public GST training
    sets, because these frames come from a published-style AIMD workflow.
    """
    labelled = read_labelled_extxyz(
        LABELLED_615,
        protocol_id=protocols.legacy_static_615().protocol_id,
        role=role,
        notes=(
            "615 frames sampled from the 216-atom AIMD runs at three number densities "
            "(0.027/0.028/0.029 at/A^3). Electronic settings match gst225_pbe_canonical_v1 "
            "except LASPH/ADDGRID/SYMPREC; PAW set is the same file (sha256 886eff9d...). "
            "Virial sign/unit must be audited before any stress metric is scored."
        ),
    )
    if decode_tags:
        decode_gst_config_types(labelled)
    return labelled


#: H5 relabel pools, keyed by the pool token in the task name:
#: ``state_kind, (T at the first source frame, T at the last)``.
H5_POOLS = {
    "md950": ("liquid_950K", (950.0, 950.0)),
    "heat": ("heating_300_1200K", (300.0, 1200.0)),
    "melt3000": ("liquid_3000K", (3000.0, 3000.0)),
    "liquid1200": ("liquid_1200K", (1200.0, 1200.0)),
    "quench": ("quench_1200_300K", (1200.0, 300.0)),
    "prod300": ("amorphous_300K", (300.0, 300.0)),
}
_H5_NAME = re.compile(r"^H5_(?P<body>[a-z0-9]+)_s(?P<seed>\d+)(?:_(?P<stage>[a-z0-9]+))?_f(?P<slot>\d+)$")


def _scf_steps(oszicar: Path) -> int:
    """Electronic steps of a single-point run, read from its OSZICAR."""
    with open(oszicar, "rb") as handle:
        return sum(1 for line in handle if line[:5] in (b"DAV: ", b"RMM: "))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def harvest_h5_labels(dest: Path = H5_LABELS) -> dict[str, object]:
    """Collect the 120 H5 static relabels into one labelled extxyz.

    Each frame keeps where it came from (source AIMD task and zero-based XDATCAR
    frame, from the task's own ``SELECTED_FRAME.json``) and the labels follow the
    project convention of ``GST225_ref/scripts/outcars2extxyz_stress.py``:
    ``REF_energy`` is TOTEN (the free energy), ``Virial`` is in eV and
    ``REF_stress = -Virial/V`` in eV/A^3.  ``REF_energy_sigma0`` is kept beside it
    because :func:`dft_relaxed` reports the sigma->0 energy.

    A task is skipped, and listed in the manifest, when its OUTCAR is incomplete
    or its SCF ran into NELM -- an unconverged label is worse than a missing one.
    """
    from ase.io import write

    canonical = protocols.canonical_static().protocol_id
    frames: list[Atoms] = []
    records: list[dict[str, object]] = []
    skipped: dict[str, str] = {}
    for task in sorted(d for d in PHASE1.glob("H5_*") if d.is_dir()):
        out = task_output_dir(task)
        outcar = out / "OUTCAR"
        if is_blocked(task) or not outcar_is_complete(outcar):
            skipped[task.name] = "no completed published run"
            continue
        nelm = int(float(read_incar(out / "INCAR").get("NELM", "60").split()[0]))
        n_scf = _scf_steps(out / "OSZICAR")
        if n_scf >= nelm:
            skipped[task.name] = f"SCF hit NELM ({n_scf}/{nelm})"
            continue
        m = _H5_NAME.match(task.name)
        if not m:
            skipped[task.name] = "task name does not follow the H5 pattern"
            continue
        pool = m.group("stage") or m.group("body")
        state_kind, (t0, t1) = H5_POOLS[pool]
        selected = json.loads((task / "SELECTED_FRAME.json").read_text())
        index = int(selected["xdatcar_frame_index_zero_based"])
        n_source = int(selected["trajectory_frames"])
        nominal_t = t0 + (t1 - t0) * index / max(n_source - 1, 1)

        final = read_outcar_final(outcar)
        source = final["atoms"]
        stress = np.asarray(final["stress"])
        frame = Atoms(
            symbols=source.get_chemical_symbols(),
            positions=source.get_positions(),
            cell=source.cell.array,
            pbc=True,
        )
        frame.info.update(
            config_type=task.name,
            REF_energy=float(source.calc.results.get("free_energy", final["energy"])),
            REF_energy_sigma0=float(final["energy"]),
            Virial=-stress * final["volume"],
            REF_stress=stress,
            protocol_id=canonical,
            source_task=selected["source_task"],
            source_frame=index,
            source_frames_total=n_source,
            pool=pool,
            state_kind=state_kind,
            seed=int(m.group("seed")),
            nominal_temperature_k=round(nominal_t, 1),
        )
        frame.arrays["REF_forces"] = np.asarray(final["forces"])
        frames.append(frame)
        records.append(
            {
                "task": task.name,
                "outcar": str(outcar),
                "outcar_sha256": _sha256(outcar),
                "scf_steps": n_scf,
                "source_task": selected["source_task"],
                "source_frame": index,
            }
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    write(str(dest), frames, format="extxyz")
    manifest = {
        "file": str(dest),
        "sha256": _sha256(dest),
        "protocol_id": canonical,
        "n_frames": len(frames),
        "skipped": skipped,
        "label_convention": "REF_energy = TOTEN; Virial in eV; REF_stress = -Virial/V (eV/A^3)",
        "tasks": records,
    }
    dest.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def labelled_h5(
    *,
    role: DataRole = DataRole.PROTECTED_AUDIT,
    path: Path | None = None,
    leakage: Path | None = None,
) -> LabelledSet:
    """The 120 H5 relabels: canonical protocol, 306 atoms at 0.0309 at/A^3.

    The first label set in this workspace computed under
    ``gst225_pbe_canonical_v1`` itself, so unlike :func:`labelled_615` it needs
    no ``d_protocol`` to be compared with other canonical numbers.

    The default role is ``protected_audit``, and it is only granted when
    ``screen_h5_leakage.py`` has screened *this exact file* (matched by sha256)
    against every GST training set in the workspace and found it clean.  Without
    that the call fails rather than quietly downgrading the role: once a frame
    has been used diagnostically it can never become protected again.

    ``path`` / ``leakage`` read the same files from somewhere else -- a reference
    pack extracted by :mod:`mlipeval.pack` -- with the same checks.
    """
    labels = Path(path) if path is not None else H5_LABELS
    screen_file = Path(leakage) if leakage is not None else H5_LEAKAGE
    if not labels.exists():
        raise FileNotFoundError(f"{labels} not found -- run `python harvest_phase1.py` first")
    if role is DataRole.PROTECTED_AUDIT:
        screen = json.loads(screen_file.read_text()) if screen_file.exists() else {}
        if screen.get("sha256") != _sha256(labels):
            raise ValueError(
                "protected_audit needs a leakage screen of the current H5 file -- run "
                "`python screen_h5_leakage.py`, or pass role=DataRole.DIAGNOSTIC knowingly"
            )
        if screen.get("verdict") != "clean":
            raise ValueError(f"H5 leakage screen is '{screen.get('verdict')}', not clean: {screen_file}")
    labelled = read_labelled_extxyz(
        labels,
        protocol_id=protocols.canonical_static().protocol_id,
        role=role,
        notes=(
            "120 static relabels of phase-1 canonical AIMD frames (306-atom Ge68Sb68Te170, "
            "0.0309 at/A^3): 30 from the 950 K production runs, 30 from the 300->1200 K "
            "heating ramps, 60 from the melt-quench chains (melt / 1200 K liquid / quench / "
            "300 K production, 5 per stage per seed)."
        ),
    )
    for key in ("pool", "state_kind", "source_task"):
        labelled.tags[key] = np.array([str(a.info[key]) for a in labelled.frames])
    for key in ("seed", "source_frame"):
        labelled.tags[key] = np.array([int(a.info[key]) for a in labelled.frames])
    labelled.tags["nominal_temperature_k"] = np.array(
        [float(a.info["nominal_temperature_k"]) for a in labelled.frames]
    )
    labelled.tags["number_density"] = labelled.n_atoms / labelled.volumes
    labelled.stress_key = "REF_stress"
    return labelled


#: The crystalline phase-1 tasks that make up the L0 diagnostic set, as
#: ``glob -> (subset, cell fixed during the run)``.  Every one of them runs the
#: canonical static electronic settings (ENCUT 500, KSPACING 0.20, EDIFF 1E-7,
#: ISYM -1), so each ionic step is a ``gst225_pbe_canonical_v1`` label.
#: Variable-cell (ISIF = 3) steps keep energy and forces but not stress: the
#: plane-wave basis is fixed at the starting cell, so their stress carries Pulay
#: error that a static at the same cell would not have.
DIAGNOSTIC_TASKS = (
    ("L1_*_static", "static_mp", True),
    ("L2_*_static", "static_element", True),
    ("H1_cubic_*_static", "static_cubic", True),
    ("H2_hex_v*", "eos_hex", True),
    ("H2_cub*_v*", "eos_cubic", True),
    ("L1_*_relax", "relax_mp", False),
    ("L2_*_relax", "relax_element", False),
    ("H1_cubic_*_relax", "relax_cubic", False),
)


def harvest_static_diagnostic(dest: Path = DIAGNOSTIC_LABELS) -> dict[str, object]:
    """Collect every ionic step of the crystalline phase-1 tasks into one extxyz.

    This is the L0 *diagnostic* set: the frames failure export and data
    selection are allowed to look at, so that the H5 ``protected_audit`` set
    never has to (``validation_methods_L0-L5.md`` section 3).  Same label
    convention as :func:`harvest_h5_labels`; ``REF_stress`` is written only for
    fixed-cell runs (see :data:`DIAGNOSTIC_TASKS`).

    Only the published run of a task is read.  ``H2_cub05_v105`` is a
    continuation whose first 160 steps live under ``attempts/``; its 3 published
    steps are what is harvested.
    """
    from ase.io import read as ase_read
    from ase.io import write

    canonical = protocols.canonical_static().protocol_id
    frames: list[Atoms] = []
    records: list[dict[str, object]] = []
    skipped: dict[str, str] = {}
    seen: set[str] = set()
    for pattern, subset, fixed_cell in DIAGNOSTIC_TASKS:
        for task in sorted(d for d in PHASE1.glob(pattern) if d.is_dir()):
            if task.name in seen:
                continue
            seen.add(task.name)
            out = task_output_dir(task)
            outcar = out / "OUTCAR"
            if is_blocked(task) or not outcar_is_complete(outcar):
                skipped[task.name] = "no completed published run"
                continue
            steps = ase_read(str(outcar), index=":", format="vasp-out")
            for k, source in enumerate(steps):
                frame = Atoms(
                    symbols=source.get_chemical_symbols(),
                    positions=source.get_positions(),
                    cell=source.cell.array,
                    pbc=True,
                )
                frame.info.update(
                    config_type=f"{task.name}:{k}",
                    REF_energy=float(source.calc.results.get("free_energy", source.get_potential_energy())),
                    REF_energy_sigma0=float(source.get_potential_energy()),
                    protocol_id=canonical,
                    source_task=task.name,
                    subset=subset,
                    ionic_step=k,
                    n_ionic_steps=len(steps),
                    fixed_cell=bool(fixed_cell),
                )
                if fixed_cell:
                    stress = np.asarray(source.get_stress(voigt=False))
                    frame.info["REF_stress"] = stress
                    frame.info["Virial"] = -stress * source.get_volume()
                frame.arrays["REF_forces"] = np.asarray(source.get_forces())
                frames.append(frame)
            records.append({"task": task.name, "subset": subset, "outcar": str(outcar),
                            "outcar_sha256": _sha256(outcar), "n_frames": len(steps)})

    dest.parent.mkdir(parents=True, exist_ok=True)
    write(str(dest), frames, format="extxyz")
    manifest = {
        "file": str(dest),
        "sha256": _sha256(dest),
        "protocol_id": canonical,
        "role": DataRole.DIAGNOSTIC.value,
        "n_frames": len(frames),
        "n_frames_with_stress": sum("REF_stress" in f.info for f in frames),
        "skipped": skipped,
        "label_convention": (
            "REF_energy = TOTEN; REF_stress (eV/A^3, ASE sign) and Virial = -REF_stress*V "
            "only for fixed-cell runs; variable-cell ISIF=3 steps carry energy and forces only"
        ),
        "tasks": records,
    }
    dest.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def labelled_diagnostic(path: Path | None = None) -> LabelledSet:
    """The L0 diagnostic set: every ionic step of the crystalline phase-1 tasks.

    Role ``diagnostic`` by construction -- this is the set failure export draws
    from.  Frames that a model's training set contains are listed per training
    set in :data:`~mlipeval.paths.DIAGNOSTIC_LEAKAGE` (``screen_diagnostic_leakage.py``);
    :func:`diagnostic_training_overlap` turns that into a per-frame mask.
    """
    labels = Path(path) if path is not None else DIAGNOSTIC_LABELS
    if not labels.exists():
        raise FileNotFoundError(f"{labels} not found -- run `python harvest_phase1.py` first")
    labelled = read_labelled_extxyz(
        labels,
        protocol_id=protocols.canonical_static().protocol_id,
        role=DataRole.DIAGNOSTIC,
        notes=(
            "crystalline phase-1 tasks under the canonical static protocol: L1 MP cells, "
            "L2 element references and H1 cubic seeds (static + every relaxation step) and "
            "the 42 H2 fixed-volume EOS relaxations. Stress only on fixed-cell frames."
        ),
    )
    for key in ("subset", "source_task"):
        labelled.tags[key] = np.array([str(a.info[key]) for a in labelled.frames])
    labelled.tags["ionic_step"] = np.array([int(a.info["ionic_step"]) for a in labelled.frames])
    labelled.stress_key = "REF_stress"
    return labelled


def diagnostic_training_overlap(
    labelled: LabelledSet,
    model_key: str,
    *,
    labels: Path | None = None,
    leakage: Path | None = None,
) -> np.ndarray:
    """Boolean mask: frames of the diagnostic set inside ``model_key``'s training set.

    Read from the screen record, which is bound to the sha256 of the harvested
    file; a stale or missing record raises instead of returning "no overlap".
    """
    from .leakage import TRAINING_SET_OF

    names = TRAINING_SET_OF.get(model_key, ())
    mask = np.zeros(len(labelled), dtype=bool)
    if not names:
        return mask
    screen_file = Path(leakage) if leakage is not None else DIAGNOSTIC_LEAKAGE
    label_file = Path(labels) if labels is not None else DIAGNOSTIC_LABELS
    if not screen_file.exists():
        raise FileNotFoundError(f"{screen_file} not found -- run `python screen_diagnostic_leakage.py`")
    screen = json.loads(screen_file.read_text())
    if screen.get("sha256") != _sha256(label_file):
        raise ValueError("diagnostic leakage screen is stale -- rerun `python screen_diagnostic_leakage.py`")
    by_config = {cfg: i for i, cfg in enumerate(labelled.config_types())}
    for name in names:
        if name not in screen["training_sets"]:
            raise KeyError(f"training set '{name}' of {model_key} was not screened")
        for cfg in screen["training_sets"][name]["overlapping_frames"]:
            mask[by_config[cfg]] = True
    return mask


def element_reference(element: str) -> Atoms:
    """Starting structure of the phase-1 L2 element reference (Materials Project ground state)."""
    return read_poscar(PHASE1 / f"L2_{element.lower()}_relax" / "POSCAR")


#: Elements of the L2 chemical-potential references.
ELEMENTS = ("Ge", "Sb", "Te")


# --------------------------------------------------------------------------
# AIMD trajectories
# --------------------------------------------------------------------------
#: The three legacy 306-atom trajectories. Their INCARs are not stored, so the
#: timestep below is the project-wide 3.0 fs *assumption*, flagged in the
#: protocol; nothing that depends on absolute time may be scored from them.
LEGACY_306 = {
    "liquid_2000K": "XDATCAR-liquid-2000K",
    "quench": "XDATCAR-quench",
    "relax_300K": "XDATCAR-relax-300K",
}


def aimd_306(
    which: str,
    *,
    stride: int = 1,
    max_frames: int | None = None,
    assumed_timestep_fs: float = 3.0,
    measure_temperature: bool = True,
) -> Trajectory:
    """One of the legacy 306-atom trajectories, tagged with its caveats.

    These files have no INCAR, so the temperature is *measured* from the
    trajectory (:mod:`mlipeval.thermo`) rather than read.  That measurement also
    validates the assumed timestep: at 3.0 fs the run named ``liquid-2000K``
    comes out at 1985 K and ``relax-300K`` at 298 K, each implying dt = 2.99 fs,
    which is what makes the project-wide 3 fs assumption defensible here.
    """
    if which not in LEGACY_306:
        raise KeyError(f"unknown 306-atom trajectory '{which}'; known: {sorted(LEGACY_306)}")
    traj = read_xdatcar(
        AIMD_306 / LEGACY_306[which],
        name=f"aimd306_{which}",
        protocol=protocols.legacy_aimd_306(),
        timestep_fs=assumed_timestep_fs,
        stride=stride,
        max_frames=max_frames,
    )
    if measure_temperature and len(traj) >= 3:
        from .thermo import attach_temperature

        attach_temperature(traj)
    return traj


#: Stage directories of the legacy 216-atom AIMD run, with complete VASP output.
LEGACY_216 = {
    "melt_2000K": "1-2000k",
    "quench_2000_300K": "2-300K",
    "relax": "3-opt",
    "amorphous_300K_a": "4-MD",
    "amorphous_300K_b": "5-MD",
}


def aimd_216(
    which: str,
    *,
    stride: int = 1,
    max_frames: int | None = None,
    measure_temperature: bool = True,
) -> Trajectory:
    """One stage of the legacy 216-atom AIMD run.

    Only INCAR/XDATCAR/POSCAR/POTCAR were kept for these runs -- there is no
    OSZICAR or OUTCAR -- so the temperature series is measured from the
    trajectory rather than read.  Note that per their own INCARs the two
    "300 K" stages are ramps (4-MD: 300->200 K, 5-MD: 300->400 K), so neither is
    an equilibrium 300 K state point.
    """
    if which not in LEGACY_216:
        raise KeyError(f"unknown 216-atom stage '{which}'; known: {sorted(LEGACY_216)}")
    traj = read_vasp_md_dir(
        AIMD_216 / LEGACY_216[which],
        protocol_id=f"gst225_legacy_aimd_216_v0:{which}",
        name=f"aimd216_{which}",
        stride=stride,
        max_frames=max_frames,
    )
    if measure_temperature and traj.temperatures is None and len(traj) >= 3:
        from .thermo import attach_temperature

        attach_temperature(traj)
    return traj


#: Phase-1 AIMD preflight segments -- short, but under the canonical protocol.
PREFLIGHT = {
    "nvt_3000K": "H0_preflight_01_nvt3000",
    "nvt_1200K": "H0_preflight_02_nvt1200",
    "nve_1200K": "H0_preflight_03_nve1200",
}


def preflight(which: str, *, stride: int = 1) -> Trajectory:
    """A phase-1 H0 preflight segment: 306 atoms, canonical AIMD protocol.

    These are the only AIMD trajectories in the workspace whose protocol is
    fully recorded, which makes them the tier-A reference for the L1 NVE drift
    check even though they are far too short for structure or transport.
    """
    if which not in PREFLIGHT:
        raise KeyError(f"unknown preflight segment '{which}'; known: {sorted(PREFLIGHT)}")
    return read_vasp_md_dir(
        PHASE1 / PREFLIGHT[which],
        protocol_id=f"gst225_aimd_canonical_v1:{which}",
        name=f"preflight_{which}",
        stride=stride,
    )


#: The phase-1 formal melt-quench chain, ``stage -> (task suffix, planned steps)``.
#: Four NVT stages per seed, 43,000 steps / 129 ps in total (TASKS.md section 2).
MELT_QUENCH_STAGES = (
    ("melt_3000K", "01_melt3000", 3_000),
    ("nvt_1200K", "02_nvt1200", 10_000),
    ("quench_1200_300K", "03_quench1200_300", 20_000),
    ("nvt_300K", "04_nvt300", 10_000),
)

MELT_QUENCH_SEEDS = (1, 2, 3)

#: Segments of a melt-quench stage that live outside the task's published run.
#: Seed 1's melt was stopped at 638 steps by the 2026-08-31 repeat-execution
#: incident; the surviving output was archived and the task was resubmitted with
#: ``NSW = 2362`` from that segment's CONTCAR.  The published run's POSCAR is
#: byte-identical to the archived CONTCAR, so the trajectory is continuous
#: (only the wavefunction restarted, ``ISTART = 0``) and the physical melt is the
#: full 3000 steps -- counting only the published run would under-report it as
#: 7.1 ps.  Each entry is ``task -> (archived XDATCAR, expected run POSCAR)``.
_ARCHIVED_PREFIXES = {
    "H4_mq_s01_01_melt3000": (
        PHASE1 / "incidents" / "20260831_h4_repeat_64906737"
        / "evidence" / "H4_mq_s01_01_melt3000",
    ),
}


def _archived_prefix_frames(task: Path) -> tuple[int, str]:
    """Frames of an archived predecessor segment, if it really is one.

    The link is only accepted when the archived ``CONTCAR`` equals the published
    run's ``POSCAR``: that is what makes the two files one trajectory rather than
    two unrelated attempts.
    """
    for archive in _ARCHIVED_PREFIXES.get(task.name, ()):
        contcar, xdatcar = archive / "CONTCAR", archive / "XDATCAR"
        poscar = task_output_dir(task) / "POSCAR"
        if not (contcar.exists() and xdatcar.exists() and poscar.exists()):
            continue
        if contcar.read_bytes() != poscar.read_bytes():
            return 0, "archived segment does not continue into the published run"
        with open(xdatcar, "rb") as handle:
            return handle.read().count(b"Direct configuration"), f"continues {archive.name} archive"
    return 0, ""


#: The phase-1 H3 chains, ``stage -> (task name template, planned steps)``.
#: 950 K: 10 ps equilibration then 15 ps production, velocities carried over.
#: Heating: one 25 ps linear 300 -> 1200 K NVT ramp per seed (~36 K/ps).
MD950_STAGES = (
    ("equil_950K", "H3_md950_s{seed:02d}_01_equil", 3_333),
    ("prod_950K", "H3_md950_s{seed:02d}_02_prod", 5_000),
)
HEATING_STAGES = (("heating_300_1200K", "H3_heat_s{seed:02d}_300_1200", 8_333),)

#: Every canonical-protocol AIMD chain, ``chain -> ((stage, task template, planned steps), ...)``.
CANONICAL_AIMD_CHAINS = {
    "H3_md950": MD950_STAGES,
    "H3_heating": HEATING_STAGES,
    "H4_melt_quench": tuple(
        (stage, f"H4_mq_s{{seed:02d}}_{suffix}", planned) for stage, suffix, planned in MELT_QUENCH_STAGES
    ),
}


def _count_frames(xdatcar: Path) -> int:
    """Frames in an XDATCAR, streamed: the H4 quench files are hundreds of MB."""
    if not xdatcar.exists():
        return 0
    with open(xdatcar, "rb") as handle:
        return sum(1 for line in handle if line.startswith(b"Direct configuration"))


def canonical_aimd_status() -> "pd.DataFrame":  # type: ignore[name-defined]
    """Per-seed, per-stage progress of every canonical-protocol AIMD chain (H3 + H4).

    ``frames`` counts the published trajectory plus any archived predecessor
    segment (:data:`_ARCHIVED_PREFIXES`), so seed 1's melt reports its physical
    3000 steps even though the published run holds 2362.
    """
    import pandas as pd

    rows: list[dict[str, object]] = []
    for chain, stages in CANONICAL_AIMD_CHAINS.items():
        for seed in MELT_QUENCH_SEEDS:
            for stage, template, planned in stages:
                task = PHASE1 / template.format(seed=seed)
                out = task_output_dir(task)
                state = (task / "TASK_STATE").read_text(encoding="utf-8", errors="replace").split()[0] \
                    if (task / "TASK_STATE").exists() else "NOT_STARTED"
                archived, note = _archived_prefix_frames(task)
                total = _count_frames(out / "XDATCAR") + archived
                rows.append(
                    {
                        "chain": chain,
                        "seed": seed,
                        "stage": stage,
                        "task": task.name,
                        "state": state,
                        "frames": total,
                        "planned_steps": planned,
                        "ps": round(total * 3.0 / 1000.0, 3),
                        "complete": total >= planned and outcar_is_complete(out / "OUTCAR"),
                        "note": note,
                    }
                )
    return pd.DataFrame(rows)


def melt_quench_status() -> "pd.DataFrame":  # type: ignore[name-defined]
    """How far the formal H4 melt-quench chain has actually got, per seed/stage.

    As of 2026-10-27 all 12 segments are complete: 3 seeds x 129 ps, every stage
    at its planned step count.  It is still *registered, not scored*: the L3/L5
    state points in :mod:`mlipeval.state_points` point at the legacy 306-atom
    trajectories, and moving them to this chain changes the density
    (0.02852 -> 0.0309 at/A^3) and the reference for every model at once, so it
    is a deliberate switch with its own MLIP MD runs, not a side effect of the
    data arriving.  Load the segments with :func:`melt_quench`.
    """
    status = canonical_aimd_status()
    return status[status["chain"] == "H4_melt_quench"].drop(columns="chain").reset_index(drop=True)


def canonical_aimd(task: str, *, stride: int = 1, max_frames: int | None = None) -> Trajectory:
    """One completed phase-1 H3/H4 AIMD segment with the canonical protocol attached.

    Refuses a segment whose OUTCAR has no final timing block: a trajectory that
    stopped early is not the state point its task name promises.

    A segment listed in :data:`_ARCHIVED_PREFIXES` is returned whole: the
    archived predecessor run first, then the published continuation.  Seed 1's
    3000 K melt is 638 archived + 2362 published steps, both starting from the
    same random-insertion POSCAR as seeds 2 and 3; read alone, the published part
    would start mid-melt and be 7.1 ps instead of 9.  The two parts are joined at
    stride 1 and only then subsampled, so the time axis stays uniform across the
    join (L4 differentiates in time).
    """
    directory = PHASE1 / task
    if not outcar_is_complete(task_output_dir(directory) / "OUTCAR"):
        raise FileNotFoundError(f"{task}: no completed published run")
    archived, _ = _archived_prefix_frames(directory)
    if not archived:
        return read_vasp_md_dir(
            directory,
            protocol_id=f"gst225_aimd_canonical_v1:{task}",
            name=task,
            stride=stride,
            max_frames=max_frames,
        )
    (archive,) = _ARCHIVED_PREFIXES[task]
    # ``max_frames`` counts frames *after* striding, as in read_vasp_md_dir
    raw = None if max_frames is None else max_frames * stride
    head = read_vasp_md_dir(archive, protocol_id=f"gst225_aimd_canonical_v1:{task}",
                            name=f"{task}[archived]", max_frames=raw)
    parts = [head]
    if raw is None or len(head) < raw:
        parts.append(read_vasp_md_dir(directory, protocol_id=f"gst225_aimd_canonical_v1:{task}", name=task,
                                      max_frames=None if raw is None else raw - len(head)))
    whole = concatenate(parts, name=task, protocol=parts[-1].protocol)
    whole = whole if stride == 1 else whole.slice(0, None, stride)
    return whole if max_frames is None else whole.slice(0, max_frames)


def _stage_task(chain: str, seed: int, stage: str) -> str:
    if seed not in MELT_QUENCH_SEEDS:
        raise KeyError(f"unknown seed {seed}; known: {MELT_QUENCH_SEEDS}")
    for name, template, _ in CANONICAL_AIMD_CHAINS[chain]:
        if name == stage:
            return template.format(seed=seed)
    known = [name for name, _, _ in CANONICAL_AIMD_CHAINS[chain]]
    raise KeyError(f"unknown {chain} stage '{stage}'; known: {known}")


def melt_quench(seed: int, stage: str, **kwargs) -> Trajectory:
    """One stage of the H4 melt-quench chain (``melt_3000K`` ... ``nvt_300K``).

    Seed 1's ``melt_3000K`` is the full 3000 steps: the 638 archived ones from
    the incident evidence followed by the published 2362 (see
    :func:`canonical_aimd`).
    """
    return canonical_aimd(_stage_task("H4_melt_quench", seed, stage), **kwargs)


def md950(seed: int, stage: str = "prod_950K", **kwargs) -> Trajectory:
    """One stage of an H3 950 K chain; the default is the 15 ps production run."""
    return canonical_aimd(_stage_task("H3_md950", seed, stage), **kwargs)


def heating(seed: int, **kwargs) -> Trajectory:
    """The H3 300 -> 1200 K heating ramp of one seed.

    A ramp, so :func:`mlipeval.l4_dynamics.assert_equilibrium_segment` must
    reject it as a diffusion reference -- it is a pathway, like the quench.
    """
    return canonical_aimd(_stage_task("H3_heating", seed, "heating_300_1200K"), **kwargs)


def amorphous_300K_production(seed: int, **kwargs) -> Trajectory:
    """The 300 K production window of one melt-quench seed: the last 15 of 30 ps.

    This is the window TASKS.md defines and the one the H5 ``prod300`` frames
    were drawn from, so it is fixed here rather than left to each caller.
    """
    return melt_quench(seed, "nvt_300K", **kwargs).last_fraction(0.5)


def preflight_nve_drift() -> dict[str, float]:
    """AIMD NVE conserved-energy drift, the tier-A target for ``L1.nve_drift``."""
    directory = task_output_dir(PHASE1 / PREFLIGHT["nve_1200K"])
    from .io_vasp import read_incar, read_outcar_md_energies

    energies = read_outcar_md_energies(directory / "OUTCAR")
    tags = read_incar(directory / "INCAR")
    dt_fs = float(tags.get("POTIM", "3.0").split()[0])
    n_atoms = len(read_poscar(directory / "POSCAR"))
    etotal = energies["etotal"]
    if etotal.size < 3:
        return {"drift_meV_per_atom_per_ps": float("nan"), "n_steps": int(etotal.size)}
    t_ps = np.arange(etotal.size) * dt_fs / 1000.0
    slope = float(np.polyfit(t_ps, etotal / n_atoms * 1000.0, 1)[0])
    return {
        "drift_meV_per_atom_per_ps": slope,
        "n_steps": int(etotal.size),
        "n_atoms": n_atoms,
        "timestep_fs": dt_fs,
        "total_time_ps": float(t_ps[-1]),
        "source": str(directory / "OUTCAR"),
        "protocol_id": "gst225_aimd_canonical_v1",
    }


# --------------------------------------------------------------------------
# structures
# --------------------------------------------------------------------------
def cubic_seed(index: int = 1) -> Atoms:
    """One of the five 194-atom cubic GST225 seeds (rocksalt with vacancies)."""
    return read_poscar(CUBIC_SEEDS / f"CONTCAR-{index}")


def mp_structure(mp_id: str) -> Atoms:
    """One of the three 9-atom Materials Project Ge2Sb2Te5 cells.

    ``mp-1224375`` and ``mp-3534`` are both P-3m1: different Te-Ge-Te stacking
    sequences of the trigonal phase, 15.13 meV/atom apart in DFT.
    ``mp-1224407`` is monoclinic Cm.  See :data:`mlipeval.l2_static.PHASE_LADDER`.
    """
    return read_poscar(MP_STRUCTURES / mp_id / "POSCAR")


def hexagonal_reference(mp_id: str = "mp-1224375") -> Atoms:
    """The phase-1 hexagonal reference: the lowest-energy of the three MP cells."""
    return mp_structure(mp_id)


def dft_relaxed(task: str) -> dict[str, object]:
    """Final geometry and energy of a completed phase-1 relaxation or static run."""
    directory = task_output_dir(PHASE1 / task)
    outcar = directory / "OUTCAR"
    if not outcar_is_complete(outcar):
        raise FileNotFoundError(f"{task}: OUTCAR missing or incomplete")
    final = read_outcar_final(outcar)
    atoms = final["atoms"]
    return {
        "task": task,
        "protocol_id": "gst225_pbe_canonical_v1",
        "atoms": atoms,
        "n_atoms": final["n_atoms"],
        "energy_eV": final["energy"],
        "energy_per_atom_eV": final["energy"] / final["n_atoms"],
        "volume_A3": final["volume"],
        "volume_per_atom_A3": final["volume"] / final["n_atoms"],
        "cell_lengths_A": np.asarray(atoms.cell.lengths()).tolist(),
        "cell_angles_deg": np.asarray(atoms.cell.angles()).tolist(),
        "max_force_eV_per_A": float(np.linalg.norm(final["forces"], axis=1).max()),
    }


def is_blocked(directory: Path) -> bool:
    """A task is still blocked only if its POSCAR was never materialised.

    ``BLOCKED`` marks a task whose POSCAR is *intentionally* absent until the
    upstream structure it depends on exists; the marker file itself is left in
    place afterwards, so its presence alone never means unfinished.

    What records the materialisation depends on which runner touched the task.
    The pre-``20260831`` runner drops a ``BLOCKED.resolved.<jobid>`` beside the
    marker; the ``20260831_execution_v1`` release instead writes
    ``MATERIALIZED_FROM`` (naming the upstream task and the transformation) next
    to the real ``POSCAR``, and three completed tasks carry only the latter.
    Testing for the marker alone silently discarded them, so both forms of
    evidence are accepted -- and the second is the substantive one, since it is
    the POSCAR's own provenance record rather than a bookkeeping side effect.
    """
    if not (directory / "BLOCKED").exists():
        return False
    if any(directory.glob("BLOCKED.resolved.*")):
        return False
    return not ((directory / "MATERIALIZED_FROM").exists() and (directory / "POSCAR").exists())


def completed_tasks(prefix: str) -> list[Path]:
    """Phase-1 task directories with the given prefix that finished cleanly."""
    return sorted(
        d
        for d in PHASE1.glob(f"{prefix}*")
        if d.is_dir() and not is_blocked(d) and outcar_is_complete(task_output_dir(d) / "OUTCAR")
    )


def dft_eos_tasks(phase: str = "hex") -> list[Path]:
    """Completed fixed-volume EOS tasks for ``hex`` or a cubic seed (``cub01`` ...).

    Accepts either the bare phase (``"hex"``, ``"cub01"``) or the full task
    prefix (``"H2_hex"``), so a caller cannot accidentally ask for
    ``H2_H2_hex_v`` and silently get an empty list back.
    """
    phase = phase.removeprefix("H2_")
    return completed_tasks(f"H2_{phase}_v")


def inventory() -> "pd.DataFrame":  # type: ignore[name-defined]
    """What reference data is actually on disk right now, as a table."""
    import pandas as pd

    rows: list[dict[str, object]] = []
    for name, filename in LEGACY_306.items():
        path = AIMD_306 / filename
        rows.append(
            {
                "dataset": f"aimd_306/{name}",
                "protocol_id": "gst225_legacy_aimd_306_v0",
                "available": path.exists(),
                "detail": "RETIRED 2026-11-02 (no INCAR stored; density 0.02852 at/A^3)",
            }
        )
    for name, directory in LEGACY_216.items():
        path = AIMD_216 / directory
        rows.append(
            {
                "dataset": f"aimd_216/{name}",
                "protocol_id": "gst225_legacy_aimd_216_v0",
                "available": (path / "XDATCAR").exists(),
                "detail": "RETIRED 2026-11-02 (SMASS=-1 velocity scaling)",
            }
        )
    for name, directory in PREFLIGHT.items():
        path = task_output_dir(PHASE1 / directory)
        rows.append(
            {
                "dataset": f"preflight/{name}",
                "protocol_id": "gst225_aimd_canonical_v1",
                "available": (path / "XDATCAR").exists(),
                "detail": "canonical protocol, but only 67-333 steps",
            }
        )
    rows.append(
        {
            "dataset": "labelled_615",
            "protocol_id": "gst225_legacy_static_v0",
            "available": LABELLED_615.exists(),
            "detail": "RETIRED 2026-11-02 (615 frames, protocol not traceable); not read by any driver",
        }
    )
    for phase in ("hex", "cub01", "cub02", "cub03", "cub04", "cub05"):
        tasks = dft_eos_tasks(phase)
        rows.append(
            {
                "dataset": f"dft_eos/{phase}",
                "protocol_id": "gst225_pbe_canonical_v1",
                "available": len(tasks) > 0,
                "detail": f"{len(tasks)} completed fixed-volume points",
            }
        )
    rows.append(
        {
            "dataset": "dft_relax/H1_cubic",
            "protocol_id": "gst225_pbe_canonical_v1",
            "available": len(completed_tasks("H1_cubic")) > 0,
            "detail": f"{len(completed_tasks('H1_cubic'))} completed relax/static tasks",
        }
    )
    for element in ("ge", "sb", "te"):
        done = [t.name for t in completed_tasks(f"L2_{element}_")]
        rows.append(
            {
                "dataset": f"dft_element_ref/{element}",
                "protocol_id": "gst225_pbe_canonical_v1",
                "available": f"L2_{element}_static" in done,
                "detail": f"{', '.join(done) if done else 'none'}; L2 chemical potentials (run_l2.py)",
            }
        )
    status = canonical_aimd_status()
    for chain, group in status.groupby("chain", sort=False):
        planned_ps = group["planned_steps"].sum() * 3.0 / 1000.0
        rows.append(
            {
                "dataset": f"dft_aimd/{chain}",
                "protocol_id": "gst225_aimd_canonical_v1",
                "available": bool(group["complete"].any()),
                "detail": (
                    f"{int(group['complete'].sum())}/{len(group)} segments complete, "
                    f"{group['ps'].sum():.1f}/{planned_ps:.1f} ps over 3 seeds"
                ),
            }
        )
    diag = DIAGNOSTIC_LABELS.with_suffix(".manifest.json")
    n_diag = json.loads(diag.read_text())["n_frames"] if diag.exists() else 0
    rows.append(
        {
            "dataset": "labelled_diagnostic",
            "protocol_id": "gst225_pbe_canonical_v1",
            "available": n_diag > 0,
            "detail": f"{n_diag} ionic steps of L1/L2/H1/H2; screened: {DIAGNOSTIC_LEAKAGE.exists()}",
        }
    )
    harvested = H5_LABELS.with_suffix(".manifest.json")
    n_h5 = json.loads(harvested.read_text())["n_frames"] if harvested.exists() else 0
    rows.append(
        {
            "dataset": "labelled_h5",
            "protocol_id": "gst225_pbe_canonical_v1",
            "available": n_h5 > 0,
            "detail": (
                f"{len(completed_tasks('H5_'))} completed relabel tasks, {n_h5} harvested into "
                f"{H5_LABELS.name}; screened: {H5_LEAKAGE.exists()}"
            ),
        }
    )
    return pd.DataFrame(rows)


#: Re-exported so a driver names the ladder in one place only.
from .l2_static import PHASE_LADDER  # noqa: E402,F401  (bottom: avoids a cycle)
