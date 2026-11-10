"""Registry of the GST225 reference protocols and the differences between them.

``database_link.md`` section 3 requires every label set to carry its own
``protocol_id`` and forbids pooling labels from different protocols.  To make
that enforceable rather than aspirational, the protocols actually present in
this workspace are declared here, together with :func:`compare_protocols`,
which reports only the *physics-relevant* tag differences -- the quantity the
bridge set has to turn into ``d_protocol``.

Traceability note (established by hashing the inputs in this workspace)
-----------------------------------------------------------------------
``GST225_reference_audit.md`` lists the DFT protocol behind the 615-frame
labelled set as untraceable.  Two facts recovered here narrow that gap:

* ``GST225_ref/POTCAR`` is byte-identical to the phase-1 canonical PAW set
  (sha256 ``886eff9d...``, ``PAW_PBE Ge 05Jan2001`` / ``Sb 06Sep2000`` /
  ``Te 08Apr2002``), so the legacy data and the new phase-1 tasks share a PAW
  dataset;
* ``GST225_ref/single_point/INCAR`` matches the canonical static protocol in
  ENCUT, KSPACING, PREC, EDIFF, ALGO, ISMEAR/SIGMA, LREAL, ISYM and NSW/IBRION.

What still differs is ``LASPH`` and ``ADDGRID`` (both switched on only in the
canonical protocol) and the explicit ``GGA = PE`` tag.  Those are small but real
energy/force effects, which is precisely what a bridge set must quantify -- so
the legacy set keeps its own ``protocol_id`` and stays diagnostic-only.
"""

from __future__ import annotations

from typing import Mapping

from .paths import GST_REF, PHASE1
from .schema import Protocol

#: Canonical PAW set shared by the legacy reference data and the phase-1 tasks.
GST_POTCAR_SHA256 = "886eff9d53dfccc9f807c4da0cfa45fdd18f2c59a0f4b69f014bb59d07e76a27"
GST_POTCAR_TITELS = ("PAW_PBE Ge 05Jan2001", "PAW_PBE Sb 06Sep2000", "PAW_PBE Te 08Apr2002")

#: INCAR tags that change energies, forces or stresses and therefore define a
#: protocol boundary.  Tags outside this list (NCORE, LWAVE, SYSTEM, ...) are
#: bookkeeping and are ignored by :func:`compare_protocols`.
PHYSICS_RELEVANT_TAGS = (
    "GGA",
    "ENCUT",
    "PREC",
    "EDIFF",
    "EDIFFG",
    "ALGO",
    "ISMEAR",
    "SIGMA",
    "KSPACING",
    "KGAMMA",
    "ISYM",
    "SYMPREC",
    "LREAL",
    "LASPH",
    "ADDGRID",
    "IVDW",
    "ISPIN",
    "LDAU",
    "NSW",
    "IBRION",
    "ISIF",
    "POTIM",
    "MDALGO",
    "SMASS",
    "ANDERSEN_PROB",
    "TEBEG",
    "TEEND",
)

#: Tags that default to a physics-changing value when absent from an INCAR.
TAG_DEFAULTS = {
    "LASPH": ".FALSE.",
    "ADDGRID": ".FALSE.",
    "ISPIN": "1",
    "IVDW": "0",
    "GGA": "PE",  # the confirmed PAW set carries LEXCH = PE
    "KGAMMA": ".TRUE.",
    "SYMPREC": "1E-5",
}


# --------------------------------------------------------------------------
# declared protocols
# --------------------------------------------------------------------------
def canonical_static() -> Protocol:
    """``gst225_pbe_canonical_v1`` -- the phase-1 static labelling protocol."""
    from .io_vasp import protocol_from_incar

    return protocol_from_incar(
        PHASE1 / "H1_cubic_s01_static/INCAR",
        "gst225_pbe_canonical_v1",
        kind="dft_static",
        description="phase-1 canonical static labels: ENCUT 500, KSPACING 0.20, EDIFF 1E-7",
        functional="PBE",
        pseudopotentials=f"potpaw_PBE.54 Ge/Sb/Te, sha256={GST_POTCAR_SHA256[:16]}",
        code_version="VASP 6.3.2 (hpccube)",
    )


def canonical_aimd() -> Protocol:
    """``gst225_aimd_canonical_v1`` -- the phase-1 306-atom AIMD protocol."""
    from .io_vasp import protocol_from_incar

    return protocol_from_incar(
        PHASE1 / "H0_preflight_02_nvt1200/INCAR",
        "gst225_aimd_canonical_v1",
        kind="aimd",
        description="phase-1 AIMD: ENCUT 300, Gamma-only, POTIM 3.0 fs, MDALGO 2 (SMASS 0)",
        functional="PBE",
        pseudopotentials=f"potpaw_PBE.54 Ge/Sb/Te, sha256={GST_POTCAR_SHA256[:16]}",
        code_version="VASP 6.3.2 (hpccube)",
    )


def legacy_static_615() -> Protocol:
    """``gst225_legacy_static_v0`` -- the protocol behind the 615 labelled frames."""
    from .io_vasp import protocol_from_incar

    return protocol_from_incar(
        GST_REF / "single_point/INCAR",
        "gst225_legacy_static_v0",
        kind="dft_static",
        description=(
            "legacy static labels for GST225_init_noStress.xyz; electronic settings match "
            "gst225_pbe_canonical_v1, but LASPH/ADDGRID are off and the run metadata "
            "(VASP version, machine, date) is not recorded"
        ),
        functional="PBE",
        pseudopotentials=f"assumed potpaw_PBE Ge/Sb/Te (GST225_ref/POTCAR sha256={GST_POTCAR_SHA256[:16]})",
        code_version="unrecorded",
    )


def legacy_aimd_216() -> Protocol:
    """``gst225_legacy_aimd_216_v0`` -- the 216-atom AIMD template INCAR."""
    from .io_vasp import protocol_from_incar

    return protocol_from_incar(
        GST_REF / "INCAR-MD",
        "gst225_legacy_aimd_216_v0",
        kind="aimd",
        description="legacy 216-atom AIMD template (ENCUT 300, POTIM 3 fs, SMASS -1 scaled)",
        functional="PBE",
        pseudopotentials=f"potpaw_PBE Ge/Sb/Te, sha256={GST_POTCAR_SHA256[:16]}",
        code_version="unrecorded",
    )


def legacy_aimd_306() -> Protocol:
    """``gst225_legacy_aimd_306_v0`` -- the 306-atom XDATCARs, no INCAR on disk.

    Only the trajectory files exist for these runs, so almost every field stays
    ``None``.  The number density that *is* measurable from the cell
    (0.02852 at/A^3) differs from the project standard 0.0309 at/A^3, which is
    recorded here because it makes these trajectories non-comparable to the
    phase-1 melt-quench chain without an explicit density statement.
    """
    return Protocol(
        protocol_id="gst225_legacy_aimd_306_v0",
        kind="aimd",
        description=(
            "306-atom liquid/quench/300 K XDATCARs in GST225_ref/AIMD/306atoms. "
            "No INCAR, OUTCAR or OSZICAR is stored alongside them, so ensemble and "
            "thermostat remain unknown; the timestep and the temperature programme are "
            "recovered from the trajectory itself (see provenance)."
        ),
        code="VASP",
        functional="PBE (assumed)",
        pseudopotentials=f"assumed potpaw_PBE Ge/Sb/Te, sha256={GST_POTCAR_SHA256[:16]}",
        timestep_fs=3.0,  # recovered, see below
        provenance={
            "timestep_source": (
                "not recorded; recovered as 2.99 fs from the kinetic temperature measured "
                "off the trajectory itself (mlipeval.thermo.infer_timestep), independently "
                "from the 2000 K liquid and the 300 K amorphous run -- consistent with the "
                "project-wide POTIM = 3.0 fs"
            ),
            "measured_mean_temperature_K": {"liquid_2000K": 1984.5, "relax_300K": 297.8, "quench": "ramp"},
            "measured_number_density_at_per_A3": 0.028524,
            "project_standard_number_density_at_per_A3": 0.0309,
            "density_mismatch_pct": round(100.0 * (0.028524 - 0.0309) / 0.0309, 2),
            "warning": (
                "8.2% below the project standard amorphous density; results from these "
                "trajectories cannot be compared against the phase-1 melt-quench chain "
                "without stating the density difference"
            ),
        },
    )


REGISTRY = {
    "gst225_pbe_canonical_v1": canonical_static,
    "gst225_aimd_canonical_v1": canonical_aimd,
    "gst225_legacy_static_v0": legacy_static_615,
    "gst225_legacy_aimd_216_v0": legacy_aimd_216,
    "gst225_legacy_aimd_306_v0": legacy_aimd_306,
}


def get(protocol_id: str) -> Protocol:
    if protocol_id not in REGISTRY:
        raise KeyError(f"unknown protocol_id '{protocol_id}'; known: {sorted(REGISTRY)}")
    return REGISTRY[protocol_id]()


# --------------------------------------------------------------------------
# protocol difference
# --------------------------------------------------------------------------
def _tags(protocol: Protocol) -> dict[str, str]:
    return dict(protocol.provenance.get("tags", {}))


#: Tags whose values must be compared numerically ("0.20" == "0.2").
_NUMERIC_TAGS = {
    "ENCUT", "EDIFF", "EDIFFG", "SIGMA", "KSPACING", "SYMPREC",
    "POTIM", "TEBEG", "TEEND", "ANDERSEN_PROB", "ISMEAR", "ISPIN",
    "ISYM", "NSW", "IBRION", "ISIF", "MDALGO", "SMASS", "IVDW",
}


#: VASP accepts several spellings of the same setting; normalise before comparing.
_ALIASES = {
    "LREAL": {"A": "AUTO", "O": "ON", ".TRUE.": "TRUE", ".T.": "TRUE", ".FALSE.": "FALSE", ".F.": "FALSE"},
    "PREC": {"N": "NORMAL", "A": "ACCURATE", "L": "LOW", "M": "MEDIUM", "H": "HIGH", "S": "SINGLE"},
}
_BOOLEAN = {".TRUE.": "TRUE", ".T.": "TRUE", "T": "TRUE", ".FALSE.": "FALSE", ".F.": "FALSE", "F": "FALSE"}


def _normalise(tag: str, value: str) -> str:
    """First token, upper-cased, with VASP's spelling variants folded together.

    VASP INCARs routinely carry a trailing free-text comment with no delimiter
    (``NSW = 4000        number of steps``), and VASP itself reads only the
    leading token -- so a naive string compare would report differences that do
    not exist.
    """
    token = str(value).strip().split()[0].upper() if str(value).strip() else "(UNSET)"
    token = _BOOLEAN.get(token, token)
    return _ALIASES.get(tag, {}).get(token, token)


def _same_value(tag: str, va: str, vb: str) -> bool:
    if tag in _NUMERIC_TAGS:
        try:
            return float(_normalise(tag, va).replace("D", "E")) == float(_normalise(tag, vb).replace("D", "E"))
        except ValueError:
            pass
    return _normalise(tag, va) == _normalise(tag, vb)


def _is_static(tags: Mapping[str, str]) -> bool:
    try:
        return int(float(tags.get("NSW", "0"))) <= 0 and int(float(tags.get("IBRION", "-1"))) == -1
    except ValueError:
        return False


def compare_protocols(a: Protocol, b: Protocol) -> dict[str, object]:
    """Physics-relevant tag differences between two VASP-derived protocols.

    The result is the *specification* of a bridge-set calculation: it says which
    settings must be varied to measure ``d_protocol``.  It deliberately does not
    estimate the size of the effect -- that requires the actual calculation.
    """
    ta, tb = _tags(a), _tags(b)
    # ISIF only selects which degrees of freedom relax; for two static runs it
    # changes nothing, so comparing it would manufacture a spurious difference.
    inert = {"ISIF"} if (_is_static(ta) and _is_static(tb)) else set()
    differences: dict[str, dict[str, str]] = {}
    for tag in PHYSICS_RELEVANT_TAGS:
        if tag in inert:
            continue
        va = ta.get(tag, TAG_DEFAULTS.get(tag, "(unset)"))
        vb = tb.get(tag, TAG_DEFAULTS.get(tag, "(unset)"))
        if not _same_value(tag, va, vb):
            differences[tag] = {a.protocol_id: _normalise(tag, va), b.protocol_id: _normalise(tag, vb)}
    same_paw = (a.pseudopotentials or "").find(GST_POTCAR_SHA256[:16]) >= 0 and (
        b.pseudopotentials or ""
    ).find(GST_POTCAR_SHA256[:16]) >= 0
    return {
        "a": a.protocol_id,
        "b": b.protocol_id,
        "n_physics_differences": len(differences),
        "differences": differences,
        "same_paw_dataset": same_paw,
        "tags_ignored_as_inert": sorted(inert),
        "mergeable": False,  # never; distinct protocol_id means distinct reference PES
        "verdict": (
            "identical physics tags; still separate protocol_ids until a bridge set confirms it"
            if not differences
            else f"{len(differences)} physics-relevant differences: {', '.join(sorted(differences))}"
        ),
        "next_step": (
            "compute a bridge set (a small shared structure list) under both protocols and "
            "report d_protocol for energy, forces and stress separately"
        ),
    }


def bridge_set_specification(a: Protocol, b: Protocol, *, n_structures: int = 30) -> dict[str, object]:
    """What a bridge set between two protocols has to contain.

    ``database_link.md`` 3.3 asks for a small shared structure list spanning
    crystal, amorphous, liquid, compressed and transition-path configurations.
    """
    diff = compare_protocols(a, b)
    return {
        "protocol_a": a.protocol_id,
        "protocol_b": b.protocol_id,
        "n_structures": n_structures,
        "coverage": [
            "cubic crystal (relaxed and strained)",
            "hexagonal crystal (relaxed)",
            "300 K amorphous",
            "1200 K liquid",
            "compressed cell (V/V0 ~ 0.92)",
            "quench-path intermediate",
        ],
        "settings_to_vary": diff["differences"],
        "reported_quantities": [
            "d_protocol(energy) after per-composition offset removal, meV/atom",
            "d_protocol(force) component RMSE, eV/A",
            "d_protocol(stress) RMSE, meV/A^3",
        ],
        "usage": "sigma_eff = max(sigma_ref, d_protocol / 2); see schema.sigma_effective",
    }
