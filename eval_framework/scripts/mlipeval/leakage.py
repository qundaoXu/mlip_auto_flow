"""Which reference data a model has already seen, and which metrics that voids.

The universal potentials in :mod:`mlipeval.models` were trained on MPtrj / sAlex
/ OMat24, none of which contains this workspace's GST225 references, so phase 1
never needed this module: a leakage screen ran once inside ``run_l0.py`` and came
back clear.

A *specialised* potential changes that.  ``mace_spec_gst`` was trained on the
public GST-ACE-24 set, which is GST chemistry by construction, and two of the
things this framework uses as references turn out to be inside it.  A number
computed on those references is not a measurement of the model -- it is a
readout of how well training converged -- and the difference has to travel with
the result rather than living in a paragraph someone may not read.

Everything below was *measured* (see :func:`screen_gst_ace24`), not assumed, and
the screen can be re-run to reproduce it.

Findings, GST-ACE-24 (4636 frames) vs this framework's references:

===================  ========================  ==================================
reference            composition matches       verdict
===================  ========================  ==================================
615-frame L0 set     0 (Ge48Sb48Te120)         clean -- nearest RDF distance 0.089
L1 binary PEC        361 dimer frames, all 6   **CONTAMINATED** -- 1.4-7.0 A
                     pairs                     covers the 1.0-8.0 A scan
L2 mp-1224375        exact match               **CONTAMINATED** -- cell identical
                                               to 0.0 A, positions to 5.6e-10 A
L2 mp-3534           291 Ge2Sb2Te5-reduced     clean -- nearest RDF distance 0.52
L2 mp-1224407        291 Ge2Sb2Te5-reduced     **CONTAMINATED** -- 18-atom supercell,
                                               RDF 0.0000 (exact-formula screen: 0.36)
L2 Ge / Sb / Te      269 / 260 / 236 frames    **CONTAMINATED** -- Ge, Sb start cells
                                               as supercells; relaxed Te
L2 cubic seeds       0 (Ge43Sb43Te108)         clean
L3/L4/L5 306-atom    0 (Ge68Sb68Te170)         clean
===================  ========================  ==================================

The 615-frame row is historical: that set was retired on 2026-11-02.  The
mp-1224407 and element rows come from the reduced-formula screen of the same
day (``screen_diagnostic_leakage.py``); see :data:`_MP1224407_IN_TRAINING`.

Composition equality is a *necessary* condition for a duplicate frame, so a
reference whose composition appears zero times cannot be in the training set at
all -- that is a stronger statement than any distance threshold, and it is why
L0, the cubic branch of L2, and all of L3/L4/L5 are reported as clean outright.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class LeakageFinding:
    """One reference set, and what it does to the metrics computed on it."""

    reference: str
    status: str            # "contaminated" | "clean"
    evidence: str
    #: Regexes matched against ``metric_id``; empty for a clean finding.
    affected_metrics: tuple[str, ...] = ()

    def matches(self, metric_id: str) -> bool:
        return any(re.match(p, metric_id) for p in self.affected_metrics)


#: The training set of ``mace_spec_gst``.
GST_ACE24 = "GST-ACE-24 (GSTACE24-VASP-PBE, 4636 frames)"

GST_ACE24_LEAKAGE: tuple[LeakageFinding, ...] = (
    LeakageFinding(
        reference="L1 binary PEC (six Ge/Sb/Te dimers, 1.0-8.0 A)",
        status="contaminated",
        evidence=(
            "GST-ACE-24 holds 361 two-atom frames covering all six pairs over "
            "1.4-7.0 A (Ge-Ge 45, Ge-Sb 36, Ge-Te 163, Sb-Sb 45, Sb-Te 36, Te-Te 36). "
            "The PEC scan range is a superset of that interval, so the curves L1 "
            "checks were directly fitted."
        ),
        affected_metrics=(r"^L1\.pec\.",),
    ),
    LeakageFinding(
        reference="L2 hexagonal reference mp-1224375 (9-atom Ge2Sb2Te5)",
        status="contaminated",
        evidence=(
            "The POSCAR this framework feeds to the MLIP relaxation is present in "
            "GST-ACE-24 as a `cryst`/`gst-gap-22` frame: cell identical to 0.0 A, "
            "sorted fractional positions to 5.6e-10, V/atom 30.343 on both sides. "
            "The DFT-relaxed geometry differs (RDF distance 0.54), so the *target* "
            "is not the trained label -- but the structure is, and mp-1224375 is the "
            "base of the phase ladder and the starting cell of the hexagonal EOS."
        ),
        affected_metrics=(
            r"^L2\.relax\.hex_mp1224375\.",
            r"^L2\.eos\.hex\.",
            r"^L2\.phase_ladder\.",
            r"^L2\.phase_order\.",
            r"^L2\.formation_energy\.dE_error\.mp1224375$",
            r"^L2\.formation_energy\.cubic_minus_hexagonal$",
        ),
    ),
    LeakageFinding(
        reference="L0 615-frame labelled set (216-atom Ge48Sb48Te120)",
        status="clean",
        evidence=(
            "zero GST-ACE-24 frames share the composition, so an identical frame is "
            "impossible; the RDF screen over the 64 same-size candidates puts the "
            "nearest neighbour at 0.089, far above the 0.02 duplicate threshold"
        ),
    ),
    LeakageFinding(
        reference="L2 cubic branch (194-atom Ge43Sb43Te108, five seeds)",
        status="clean",
        evidence="zero GST-ACE-24 frames share the composition",
    ),
    LeakageFinding(
        reference="L2 mp-3534 and mp-1224407 (9-atom Ge2Sb2Te5)",
        status="clean",
        evidence=(
            "34 GST-ACE-24 frames share the composition, but the nearest is 0.52 "
            "(mp-3534) and 0.36 (mp-1224407) away in RDF fingerprint -- distinct "
            "structures, not the same cell"
        ),
    ),
    LeakageFinding(
        reference="L3/L4/L5 306-atom trajectories (Ge68Sb68Te170)",
        status="clean",
        evidence="zero GST-ACE-24 frames share the composition",
    ),
)

#: The training set of ``mace_spec_gst_ne``.
PROCESSED_NE = "processed_NE (PROCNE-VASP-PBE, 2528 frames)"

#: Measured by :func:`screen_processed_ne` on 2026-09-08.  Note this profile is
#: *narrower* than GST-ACE-24's: processed_NE holds no two-atom frames at all
#: (its smallest frame is 3 atoms), so the L1 PEC -- and therefore the L1 gate --
#: is clean here, while it is contaminated for ``mace_spec_gst``.
PROCESSED_NE_LEAKAGE: tuple[LeakageFinding, ...] = (
    LeakageFinding(
        reference="L2 hexagonal reference mp-1224375 (9-atom Ge2Sb2Te5)",
        status="contaminated",
        evidence=(
            "The POSCAR this framework feeds to the MLIP relaxation is present in "
            "processed_NE as a `GST-NE` frame: of 30 same-composition candidates one "
            "matches exactly -- cell identical to 0.0 A, sorted fractional positions "
            "to 7.5e-7, V/atom 30.3428 on both sides (REF_energy -33.96188909). "
            "Same reference that leaks into GST-ACE-24, so this contamination is "
            "common to both specialised MACE potentials."
        ),
        affected_metrics=(
            r"^L2\.relax\.hex_mp1224375\.",
            r"^L2\.eos\.hex\.",
            r"^L2\.phase_ladder\.",
            r"^L2\.phase_order\.",
            r"^L2\.formation_energy\.dE_error\.mp1224375$",
            r"^L2\.formation_energy\.cubic_minus_hexagonal$",
        ),
    ),
    LeakageFinding(
        reference="L1 binary PEC (six Ge/Sb/Te dimers, 1.0-8.0 A)",
        status="clean",
        evidence=(
            "processed_NE contains no two-atom frames at all -- its smallest frame is "
            "3 atoms -- so none of the six pair curves can have been fitted. This is "
            "the key difference from GST-ACE-24, where all six PECs are contaminated; "
            "the L1 gate is therefore scoreable for this model."
        ),
    ),
    LeakageFinding(
        reference="L0 615-frame labelled set (216-atom Ge48Sb48Te120)",
        status="clean",
        evidence="zero processed_NE frames share the composition, so an identical frame is impossible",
    ),
    LeakageFinding(
        reference="L2 cubic branch (194-atom Ge43Sb43Te108, five seeds)",
        status="clean",
        evidence="zero processed_NE frames share the composition",
    ),
    LeakageFinding(
        reference="L2 mp-3534 and mp-1224407 (9-atom Ge2Sb2Te5)",
        status="clean",
        evidence=(
            "30 processed_NE frames share the composition, but the nearest is 0.53 "
            "(mp-3534) and 0.36 (mp-1224407) away in RDF fingerprint -- distinct "
            "structures, not the same cell"
        ),
    ),
    LeakageFinding(
        reference="L3/L4/L5 306-atom trajectories (Ge68Sb68Te170)",
        status="clean",
        evidence="zero processed_NE frames share the composition",
    ),
)


#: The training set of ``mace_gst_aging`` (4123 train + 454 val frames).
GST_AGING = "GST225-aging (GST225-{train,val}-updated.xyz, 4577 frames)"

#: Measured 2026-09-08.  Same single contamination as processed_NE, but this set
#: additionally *does* contain the L3/L4/L5 composition -- which screened clean.
GST_AGING_LEAKAGE: tuple[LeakageFinding, ...] = (
    LeakageFinding(
        reference="L2 hexagonal reference mp-1224375 (9-atom Ge2Sb2Te5)",
        status="contaminated",
        evidence=(
            "30 same-composition candidates, one an exact hit (RDF distance 0.000000). "
            "The same reference leaks into GST-ACE-24 and processed_NE, so this "
            "contamination is common to every GST-specialised model here."
        ),
        affected_metrics=(
            r"^L2\.relax\.hex_mp1224375\.",
            r"^L2\.eos\.hex\.",
            r"^L2\.phase_ladder\.",
            r"^L2\.phase_order\.",
            r"^L2\.formation_energy\.dE_error\.mp1224375$",
            r"^L2\.formation_energy\.cubic_minus_hexagonal$",
        ),
    ),
    LeakageFinding(
        reference="L3/L4/L5 306-atom trajectories (Ge68Sb68Te170)",
        status="clean",
        evidence=(
            "Unlike GST-ACE-24 and processed_NE, this set DOES contain the composition: "
            "70 frames, all config_type `GST-amor-init`, V/atom 33.0-42.0 (median 39.0) "
            "against the reference's 35.06. Screened against all three legacy "
            "trajectories at stride 10, the nearest RDF distance is 0.064 "
            "(liquid_2000K), 0.106 (quench) and 0.083 (relax_300K) -- all far above the "
            "0.02 duplicate threshold, i.e. same composition but distinct structures."
        ),
    ),
    LeakageFinding(
        reference="L1 binary PEC (six Ge/Sb/Te dimers, 1.0-8.0 A)",
        status="clean",
        evidence="no two-atom frames in the training or validation set",
    ),
    LeakageFinding(
        reference="L0 615-frame labelled set (216-atom Ge48Sb48Te120)",
        status="clean",
        evidence="zero frames share the composition",
    ),
    LeakageFinding(
        reference="L2 cubic branch (194-atom Ge43Sb43Te108, five seeds)",
        status="clean",
        evidence="zero frames share the composition",
    ),
    LeakageFinding(
        reference="L2 mp-3534 and mp-1224407 (9-atom Ge2Sb2Te5)",
        status="clean",
        evidence="30 same-composition frames, nearest RDF 0.53 and 0.36 -- distinct structures",
    ),
)



AL_LADDER_CLEAN = "GST225 AL ladder, generations gen0-gen9"
AL_PROD = "GST225 AL ladder, prod_train (3864 frames after de-dup)"

#: Measured by ``screen_al_ladder.py`` + ``screen_al_ladder_306.py`` on
#: 2026-09-14, one generation at a time. The ladder points are NOT nested --
#: prod_train drops 151 of gen9's frames -- so inheriting the ``mace_gst_aging``
#: screen would have been an assumption, and it would have been the wrong one:
#: the hexagonal contamination enters only at the final production step.
AL_LADDER_LEAKAGE: tuple[LeakageFinding, ...] = (
    LeakageFinding(
        reference="L2 hexagonal reference mp-1224375 (9-atom Ge2Sb2Te5)",
        status="clean",
        evidence=(
            "zero frames share the composition Ge2Sb2Te5 in any of gen0-gen9. "
            "The 9-atom hexagonal cell enters the lineage only at prod_train, "
            "so the L2 phase ladder is measurable for every AL generation and "
            "only voided at the top of the ladder (see AL_PROD_LEAKAGE)."
        ),
    ),
    LeakageFinding(
        reference="L3/L4/L5 306-atom trajectories (Ge68Sb68Te170)",
        status="clean",
        evidence=(
            "gen0/gen3/gen4 contain no 306-atom frame at all; 306-atom cells "
            "enter at gen6 (112 frames, V/atom 40.69), drop to 55 at gen8-gen9 "
            "(38.27). Screened against all three legacy trajectories at stride "
            "10, nearest RDF distance 0.0635 (liquid_2000K), 0.107 (quench), "
            "0.083 (relax_300K) -- same composition, distinct structures, and "
            "reproducing the numbers recorded for mace_gst_aging."
        ),
    ),
    LeakageFinding(
        reference="L1 binary PEC (six Ge/Sb/Te dimers, 1.0-8.0 A)",
        status="clean",
        evidence="no two-atom frames in any generation",
    ),
    LeakageFinding(
        reference="L0 615-frame labelled set (216-atom Ge48Sb48Te120)",
        status="clean",
        evidence="zero frames share the composition, in every generation",
    ),
    LeakageFinding(
        reference="L2 cubic branch (194-atom Ge43Sb43Te108, five seeds)",
        status="clean",
        evidence="zero frames share the composition, in every generation",
    ),
    LeakageFinding(
        reference="L2 mp-3534 and mp-1224407 (9-atom Ge2Sb2Te5)",
        status="clean",
        evidence="zero frames share the composition in gen0-gen9",
    ),
)

#: prod_train differs from the generations by exactly one finding, and it is the
#: one that matters for the top of the ladder.
AL_PROD_LEAKAGE: tuple[LeakageFinding, ...] = (
    LeakageFinding(
        reference="L2 hexagonal reference mp-1224375 (9-atom Ge2Sb2Te5)",
        status="contaminated",
        evidence=(
            "30 same-composition candidates, one an exact hit (RDF distance "
            "0.000000) -- the same leak mace_gst_aging carries, which is "
            "expected: prod_train is that model's training set after de-dup."
        ),
        affected_metrics=(
            r"^L2\.relax\.hex_mp1224375\.",
            r"^L2\.eos\.hex\.",
            r"^L2\.phase_ladder\.",
            r"^L2\.phase_order\.",
            r"^L2\.formation_energy\.dE_error\.mp1224375$",
            r"^L2\.formation_energy\.cubic_minus_hexagonal$",
        ),
    ),
) + AL_LADDER_LEAKAGE[1:]


#: Measured 2026-11-02 by ``screen_diagnostic_leakage.py``, which matches on the
#: *reduced* formula.  The screens above matched the exact formula and so could
#: not see a supercell of a reference cell -- which is how mp-1224407 and the
#: element ground states sit in these training sets.  The earlier "mp-1224407
#: clean (nearest 0.36)" verdict was that blind spot, and is superseded here.
_MP1224407_IN_TRAINING = LeakageFinding(
    reference="L2 monoclinic mp-1224407 (9-atom Ge2Sb2Te5)",
    status="contaminated",
    evidence=(
        "present as an 18-atom `cryst` supercell: RDF distance 0.0000, V/atom 31.908 on both "
        "sides. The exact-formula screen compared only 9-atom frames (nearest 0.36) and missed it. "
        "Same hit in GST-ACE-24, processed_NE, GST225-aging and AL prod_train."
    ),
    affected_metrics=(r"^L2\.relax\.phase_mp1224407\.", r"^L2\.formation_energy\.dE_error\.mp1224407$"),
)
_MP3534_CLEAN = LeakageFinding(
    reference="L2 mp-3534 (9-atom Ge2Sb2Te5)",
    status="clean",
    evidence="no frame of any cell size within RDF 0.02 (reduced-formula screen, 2026-11-02)",
)
_ELEMENTS_IN_TRAINING = LeakageFinding(
    reference="L2 element references Ge / Sb / Te",
    status="contaminated",
    evidence=(
        "the Ge and Sb ground-state cells the MLIP relaxation starts from are present as "
        "supercells (Ge 8/64 atoms, Sb 6/48 atoms, RDF 0.0000), and the DFT-relaxed Te "
        "(L2_te_static, the last relaxation steps) matches a training frame. The chemical "
        "potentials therefore come from trained structures, which voids every formation energy."
    ),
    affected_metrics=(r"^L2\.relax\.element_", r"^L2\.formation_energy\."),
)


def _supersede_reduced_formula(findings: tuple[LeakageFinding, ...]) -> tuple[LeakageFinding, ...]:
    kept = tuple(f for f in findings if not f.reference.startswith("L2 mp-3534 and mp-1224407"))
    return kept + (_MP3534_CLEAN, _MP1224407_IN_TRAINING, _ELEMENTS_IN_TRAINING)


GST_ACE24_LEAKAGE = _supersede_reduced_formula(GST_ACE24_LEAKAGE)
PROCESSED_NE_LEAKAGE = _supersede_reduced_formula(PROCESSED_NE_LEAKAGE)
GST_AGING_LEAKAGE = _supersede_reduced_formula(GST_AGING_LEAKAGE)
AL_PROD_LEAKAGE = _supersede_reduced_formula(AL_PROD_LEAKAGE)
AL_LADDER_LEAKAGE = tuple(f for f in AL_LADDER_LEAKAGE if not f.reference.startswith("L2 mp-3534 and mp-1224407")) + (
    LeakageFinding(
        reference="L2 MP cells and element references (reduced-formula screen)",
        status="clean",
        evidence=(
            "gen0-gen9 hold Ge2Sb2Te5-reduced supercells (422-1161 frames) but the nearest is "
            "0.74-0.83 in RDF fingerprint; no Ge, Sb or Te element frame at all (2026-11-02)"
        ),
    ),
)

#: Per model key, the training set and its findings.
LEAKAGE: dict[str, tuple[str, tuple[LeakageFinding, ...]]] = {
    "mace_spec_gst": (GST_ACE24, GST_ACE24_LEAKAGE),
    # Same training set, same split (prepare_datasets.py, seed 20260902), so the
    # screen done for mace_spec_gst transfers verbatim -- it is a property of the
    # data against this framework's references, not of the architecture.
    "nequip_spec_gst": (GST_ACE24, GST_ACE24_LEAKAGE),
    "mace_spec_gst_ne": (PROCESSED_NE, PROCESSED_NE_LEAKAGE),
    "mace_gst_aging": (GST_AGING, GST_AGING_LEAKAGE),
    **{k: (AL_LADDER_CLEAN, AL_LADDER_LEAKAGE)
       for k in ("al_gen0", "al_gen3", "al_gen4", "al_gen6", "al_gen7",
                 "al_gen8", "al_gen9")},
    "al_prod": (AL_PROD, AL_PROD_LEAKAGE),
}


#: Training-set files by screen name, shared by the frame-level screens
#: (``screen_h5_leakage.py``, ``screen_diagnostic_leakage.py``).  Resolved
#: lazily: :mod:`mlipeval.paths` imports nothing from here.
def _training_set_files() -> dict[str, list[Path]]:
    from . import paths

    specialised = paths.WORKSPACE / "MLIP_run/specialized"
    aging = Path("/scratch/t0937929/myproject/1-GST225-aging/dataset")
    ladder = specialised / "al_ladder/train_sets"
    return {
        "GST-ACE-24": [paths.EXTERNAL_ACE],
        "processed_NE": [paths.EXTERNAL_NE],
        "GST225-aging": [aging / "GST225-train-updated.xyz", aging / "GST225-val-updated.xyz"],
        **{f"AL {p.stem}": [p] for p in sorted(ladder.glob("[!.]*.xyz"))},
    }


TRAINING_SET_FILES = _training_set_files()

#: Which screened training set(s) each specialised model was fitted on.  The
#: ``gst_*`` / ``gst_ne_*`` splits are subsets of GST-ACE-24 / processed_NE, so
#: the parent set is the conservative screen.  Universal potentials (MPtrj /
#: sAlex / OMat24) are absent: none of those sets is in this workspace.
TRAINING_SET_OF: dict[str, tuple[str, ...]] = {
    "mace_spec_gst": ("GST-ACE-24",),
    "nequip_spec_gst": ("GST-ACE-24",),
    "mace_spec_gst_ne": ("processed_NE",),
    "mace_gst_aging": ("GST225-aging",),
    **{f"al_{g}": (f"AL {g}",) for g in ("gen0", "gen3", "gen4", "gen6", "gen7", "gen8", "gen9")},
    "al_prod": ("AL prod_train",),
}


def findings_for(model_key: str) -> tuple[LeakageFinding, ...]:
    """Leakage findings recorded for one model; empty when nothing is known."""
    entry = LEAKAGE.get(model_key)
    return entry[1] if entry else ()


def contaminated(model_key: str, metric_id: str) -> LeakageFinding | None:
    """The finding that voids ``metric_id`` for this model, if any."""
    for finding in findings_for(model_key):
        if finding.status == "contaminated" and finding.matches(metric_id):
            return finding
    return None


def apply_to_results(results, model_key: str) -> dict[str, int]:
    """Stamp leakage onto :class:`~mlipeval.schema.MetricResult` objects in place.

    Called from :meth:`mlipeval.schema.ResultSet.save`, which is the one place
    every driver goes through, so a specialised model cannot produce a result
    file whose contaminated entries are unmarked -- not even from a driver
    written after this module.

    A contaminated metric that was ``ok`` is downgraded to ``provisional``: a
    number the model was trained on must not read as a passed test.
    """
    training_set = LEAKAGE.get(model_key, ("", ()))[0]
    counts = {"annotated": 0, "downgraded": 0}
    for record in results:
        finding = contaminated(model_key, getattr(record, "metric_id", ""))
        if finding is None:
            continue
        record.extra["training_leakage"] = {
            "training_set": training_set,
            "reference": finding.reference,
            "evidence": finding.evidence,
            "meaning": (
                "the model was trained on this reference; the number measures "
                "training convergence, not generalisation"
            ),
        }
        counts["annotated"] += 1
        status = getattr(record, "status", None)
        if status is not None and getattr(status, "value", status) == "ok":
            from .schema import Status

            record.status = Status.PROVISIONAL
            record.failure_reason = (record.failure_reason or None)
            counts["downgraded"] += 1
    return counts


def context_for(model_key: str) -> dict[str, object] | None:
    """The screen summary a result file should carry in its ``context``."""
    if model_key not in LEAKAGE:
        return None
    return {
        "model": model_key,
        "training_set": LEAKAGE[model_key][0],
        "contaminated_references": [
            f.reference for f in findings_for(model_key) if f.status == "contaminated"
        ],
        "clean_references": [
            f.reference for f in findings_for(model_key) if f.status == "clean"
        ],
        "note": (
            "metrics computed on a contaminated reference carry extra.training_leakage "
            "and are never reported as 'ok'"
        ),
    }


def annotate_result_file(path: str | Path, model_key: str) -> dict[str, int]:
    """Stamp ``training_leakage`` into every affected metric of a result file.

    Rewrites the file in place, adding to each contaminated metric's ``extra``:

    ``{"training_leakage": {"training_set": ..., "reference": ..., "evidence": ...}}``

    and setting ``status`` to ``"provisional"`` if it was ``"ok"`` -- a number the
    model was trained on must not read as a passed test.  Returns a count per
    status so a driver can report what it touched.
    """
    import json

    path = Path(path)
    payload = json.loads(path.read_text())
    training_set = LEAKAGE.get(model_key, ("", ()))[0]
    counts = {"annotated": 0, "downgraded": 0}
    for record in payload.get("results", []):
        finding = contaminated(model_key, record.get("metric_id", ""))
        if finding is None:
            continue
        extra = record.setdefault("extra", {})
        extra["training_leakage"] = {
            "training_set": training_set,
            "reference": finding.reference,
            "evidence": finding.evidence,
            "meaning": (
                "the model was trained on this reference; the number measures "
                "training convergence, not generalisation"
            ),
        }
        counts["annotated"] += 1
        if record.get("status") == "ok":
            record["status"] = "provisional"
            counts["downgraded"] += 1
    payload.setdefault("context", {})["training_leakage_screen"] = {
        "model": model_key,
        "training_set": training_set,
        "contaminated_references": [
            f.reference for f in findings_for(model_key) if f.status == "contaminated"
        ],
        "clean_references": [
            f.reference for f in findings_for(model_key) if f.status == "clean"
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return counts


# --------------------------------------------------------------------------
# the screen itself, so the table above is reproducible rather than asserted
# --------------------------------------------------------------------------
def _screen_external(path, label: str) -> "pd.DataFrame":  # type: ignore[name-defined]
    """Shared screening core: composition match first, then RDF fingerprint.

    Composition equality is a *necessary* condition for a duplicate frame, so a
    zero composition count is a stronger statement than any distance threshold.
    """
    import numpy as np
    import pandas as pd
    from ase.io import read

    from . import datasets
    from .neighbors import total_rdf_fingerprint

    ext = read(str(path), index=":")
    rows: list[dict[str, object]] = []

    def comp(atoms) -> str:
        return atoms.get_chemical_formula(mode="hill")

    ext_by_comp: dict[str, list] = {}
    for a in ext:
        ext_by_comp.setdefault(comp(a), []).append(a)

    for mp_id in ("mp-1224375", "mp-3534", "mp-1224407"):
        ref = datasets.mp_structure(mp_id)
        cands = ext_by_comp.get(comp(ref), [])
        if not cands:
            rows.append({"training_set": label, "reference": mp_id,
                         "composition_matches": 0, "nearest_rdf": np.nan,
                         "verdict": "clean (no composition match)"})
            continue
        fp = total_rdf_fingerprint(ref, r_max=6.0, bins=60)
        d = np.array([np.linalg.norm(total_rdf_fingerprint(a, r_max=6.0, bins=60) - fp)
                      for a in cands])
        rows.append({"training_set": label, "reference": mp_id,
                     "composition_matches": len(cands), "nearest_rdf": float(d.min()),
                     "verdict": "CONTAMINATED" if d.min() < 0.02 else "clean"})

    for name, formula in (("L0 615-frame set", "Ge48Sb48Te120"),
                          ("L2 cubic seeds", "Ge43Sb43Te108"),
                          ("L3/L4/L5 306-atom", "Ge68Sb68Te170")):
        n = len(ext_by_comp.get(formula, []))
        rows.append({"training_set": label, "reference": f"{name} ({formula})",
                     "composition_matches": n, "nearest_rdf": np.nan,
                     "verdict": "clean (no composition match)" if n == 0 else "inspect"})

    dimers = [a for a in ext if len(a) == 2]
    if not dimers:
        rows.append({"training_set": label, "reference": "L1 PEC (all six pairs)",
                     "composition_matches": 0, "nearest_rdf": np.nan,
                     "verdict": "clean (no two-atom frames at all)"})
    else:
        pairs: dict[tuple[str, str], list[float]] = {}
        for a in dimers:
            key = tuple(sorted(a.get_chemical_symbols()))  # type: ignore[assignment]
            pairs.setdefault(key, []).append(a.get_distance(0, 1, mic=True))
        for key, ds in sorted(pairs.items()):
            arr = np.array(ds)
            rows.append({"training_set": label,
                         "reference": f"L1 PEC {'-'.join(key)}",
                         "composition_matches": len(ds), "nearest_rdf": np.nan,
                         "verdict": f"CONTAMINATED ({arr.min():.2f}-{arr.max():.2f} A)"})
    return pd.DataFrame(rows)


def screen_processed_ne() -> "pd.DataFrame":  # type: ignore[name-defined]
    """Re-derive the processed_NE findings from the file on disk.

    Same screen as :func:`screen_gst_ace24`, run against the training set of
    ``mace_spec_gst_ne``.  Exists for the same reason: so the constants above can
    be checked rather than believed.
    """
    from . import paths
    return _screen_external(paths.EXTERNAL_NE, PROCESSED_NE)


def screen_gst_ace24() -> "pd.DataFrame":  # type: ignore[name-defined]
    """Re-derive the GST-ACE-24 findings from the files on disk.

    Slow (it reads 4636 frames and builds RDF fingerprints), so it is not called
    by any driver; it exists so the constants above can be checked rather than
    believed.
    """
    import numpy as np
    import pandas as pd
    from ase.io import read

    from . import datasets, paths
    from .neighbors import total_rdf_fingerprint

    ext = read(str(paths.EXTERNAL_ACE), index=":")
    rows: list[dict[str, object]] = []

    def comp(atoms) -> str:
        return atoms.get_chemical_formula(mode="hill")

    ext_by_comp: dict[str, list] = {}
    for a in ext:
        ext_by_comp.setdefault(comp(a), []).append(a)

    # crystalline references
    for mp_id in ("mp-1224375", "mp-3534", "mp-1224407"):
        ref = datasets.mp_structure(mp_id)
        cands = ext_by_comp.get(comp(ref), [])
        if not cands:
            rows.append({"reference": mp_id, "composition_matches": 0,
                         "nearest_rdf": np.nan, "verdict": "clean (no composition match)"})
            continue
        fp = total_rdf_fingerprint(ref, r_max=6.0, bins=60)
        d = np.array([np.linalg.norm(total_rdf_fingerprint(a, r_max=6.0, bins=60) - fp)
                      for a in cands])
        rows.append({"reference": mp_id, "composition_matches": len(cands),
                     "nearest_rdf": float(d.min()),
                     "verdict": "CONTAMINATED" if d.min() < 0.02 else "clean"})

    # the compositions that simply do not occur
    for name, formula in (("L0 615-frame set", "Ge48Sb48Te120"),
                          ("L2 cubic seeds", "Ge43Sb43Te108"),
                          ("L3/L4/L5 306-atom", "Ge68Sb68Te170")):
        n = len(ext_by_comp.get(formula, []))
        rows.append({"reference": f"{name} ({formula})", "composition_matches": n,
                     "nearest_rdf": np.nan,
                     "verdict": "clean (no composition match)" if n == 0 else "inspect"})

    # dimers vs the PEC scan
    dimers = [a for a in ext if len(a) == 2]
    pairs: dict[tuple[str, str], list[float]] = {}
    for a in dimers:
        key = tuple(sorted(a.get_chemical_symbols()))  # type: ignore[assignment]
        pairs.setdefault(key, []).append(a.get_distance(0, 1, mic=True))
    for key, ds in sorted(pairs.items()):
        arr = np.array(ds)
        rows.append({"reference": f"L1 PEC {'-'.join(key)}", "composition_matches": len(ds),
                     "nearest_rdf": np.nan,
                     "verdict": f"CONTAMINATED ({arr.min():.2f}-{arr.max():.2f} A)"})
    return pd.DataFrame(rows)
