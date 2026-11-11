"""Rescore L0/L1/L2 from an evaluation pack, on a CPU, without the model.

Every number here is produced by the same functions the HPC drivers call
(``evaluate_l0``, the ``l1_pes`` judges, the ``l2_static`` metrics); only the
model evaluations are replaced by what the pack carries.  The output is a set
of ordinary :class:`~mlipeval.schema.ResultSet` files, so the ranking, report
card and notebooks read them like any other.

:func:`compare_results` checks a rescored file against the HPC original that
rode along in the pack; for a pack built from current outputs every value,
status and grade should agree (backfilled L2 cell lengths excepted, which the
result file stored rounded).
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .pack import ModelPack, PackError, Reference, verify_model_pack
from .schema import ResultSet

LEVELS = ("L0", "L1", "L2")
RESCORE_VERSION = "rescore/1"

L2_NOTE = (
    "every comparison is provisional (validation_methods_L0-L5.md section 4). "
    "The cubic EOS now has five DFT references, but they are five vacancy "
    "orderings rather than five repeats, so their spread is configurational and "
    "is not usable as sigma_ref; the hexagonal side still has a single reference."
)


@dataclass
class Rescored:
    """Result sets keyed by file stem (``L0_audit``, ``L1``, ...), plus plot data."""

    pack: ModelPack
    reference: Reference
    verification: dict[str, Any]
    results: dict[str, ResultSet] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    plots: dict[str, Any] = field(default_factory=dict)

    def file_name(self, stem: str) -> str:
        if stem in ("L1", "L2"):
            return f"{stem}_{self.pack.model_key}.json"
        return f"{stem}.json"

    def save(self, out_dir: str | Path) -> dict[str, Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written = {}
        for stem, rs in self.results.items():
            written[stem] = rs.save(out_dir / self.file_name(stem), model_key=self.pack.model_key)
        (out_dir / "rescore_summary.json").write_text(
            json.dumps(self.summary(), indent=2, ensure_ascii=False, default=float) + "\n", encoding="utf-8")
        return written

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "pack_id": self.pack.pack_id,
            "model_key": self.pack.model_key,
            "reference_id": self.reference.reference_id,
            "engine": RESCORE_VERSION,
            "verification": {k: self.verification[k] for k in ("ok", "n_errors", "n_warnings", "blocked_levels")},
            "skipped": self.skipped,
            "results": {},
        }
        for stem, rs in self.results.items():
            statuses: dict[str, int] = {}
            for r in rs:
                statuses[r.status.value] = statuses.get(r.status.value, 0) + 1
            entry: dict[str, Any] = {"n_metrics": len(rs), "status_counts": statuses}
            if stem == "L1":
                entry["gate"] = rs.context.get("gate")
            out["results"][stem] = entry
        return out


def _stamp(rs: ResultSet, pack: ModelPack, ref: Reference, warnings: Iterable[dict]) -> None:
    from . import leakage

    # the same training-leakage pass ResultSet.save applies, done now so the
    # in-memory results (comparison, Workbench view) match the written file;
    # it is idempotent, so the second pass at save time changes nothing
    leakage.apply_to_results(rs.results, pack.model_key)
    context = leakage.context_for(pack.model_key)
    if context is not None:
        rs.context["training_leakage_screen"] = context
    rs.context["rescored_from_pack"] = {
        "engine": RESCORE_VERSION,
        "pack_id": pack.pack_id,
        "pack_content_sha256": pack.manifest.get("content_sha256"),
        "reference_id": ref.reference_id,
        "reference_content_sha256": ref.manifest.get("content_sha256"),
        "rescored_at": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
        "warnings": [w["detail"] for w in warnings],
        "backfilled": pack.manifest.get("provenance", {}).get("backfilled", []),
    }


# ==========================================================================
# L0
# ==========================================================================
def rescore_l0(pack: ModelPack, ref: Reference) -> tuple[dict[str, ResultSet], dict[str, Any]]:
    from .cache import load_predictions
    from .l0_reference import evaluate_l0

    spec = pack.spec
    level = pack.level("L0")
    out: dict[str, ResultSet] = {}
    plots: dict[str, Any] = {}

    if "audit" in level:
        labelled = ref.h5()
        screen = ref.h5_leakage()
        predictions = load_predictions(pack.path(level["audit"]["file"]))
        report = evaluate_l0(
            labelled, spec, predictions=predictions,
            tags={k: labelled.tags[k] for k in ("pool", "state_kind")},
            leakage_note=(f"H5 screened '{screen['verdict']}' against every GST training set in the "
                          f"workspace (sha256 {screen['sha256'][:12]}, threshold {screen['threshold_rdf']})"),
        )
        report.results.context["failure_export"] = "never run on this set: it is protected_audit"
        report.results.context["reference_source"] = f"{ref.reference_id}:L0/h5_audit.extxyz"
        out["L0_audit"] = report.results
        plots["audit"] = _l0_plot(report, labelled)

    if "diagnostic" in level:
        labelled = ref.diagnostic()
        frames = pack.json(level["diagnostic"]["frames"])
        mask = np.zeros(len(labelled), dtype=bool)
        mask[np.asarray(frames["kept_indices"], dtype=int)] = True
        removed = int(frames["training_overlap_removed"])
        labelled = labelled.subset(mask)
        predictions = load_predictions(pack.path(level["diagnostic"]["file"]))
        report = evaluate_l0(
            labelled, spec, predictions=predictions,
            tags={"subset": labelled.tags["subset"]},
            leakage_note=(f"{removed} frame(s) inside this model's training set removed before "
                          "scoring (screen_diagnostic_leakage.py)"),
        )
        report.results.context["training_overlap_removed"] = removed
        report.results.context["candidate_manifest"] = None
        report.results.context["failure_export"] = (
            "not run when rescoring: candidate de-duplication needs model descriptors; "
            "export candidates from the HPC run (`mlipeval run l0`)")
        report.results.context["reference_source"] = f"{ref.reference_id}:L0/diagnostic.extxyz"
        out["L0_diagnostic"] = report.results
        plots["diagnostic"] = _l0_plot(report, labelled)
    return out, plots


def _l0_plot(report, labelled, *, n_parity: int = 700, seed: int = 0) -> dict[str, Any]:
    """Plot-ready arrays in the Workbench shape (``models[k].l0``)."""
    rng = np.random.default_rng(seed)
    n = report.force_ref_flat.size
    take = np.sort(rng.choice(n, size=min(n_parity, n), replace=False)) if n else np.zeros(0, int)
    ref_forces = labelled.forces()
    frame_err, frame_max = [], []
    for fr, fm in zip(ref_forces, report.predictions.forces):
        err = np.linalg.norm(fm - fr, axis=1)  # per-atom |dF|, as the page plots it
        frame_err.append(float(np.sqrt(np.mean(err ** 2))))
        frame_max.append(float(np.max(err)))
    return {
        "parity": [np.round(report.force_ref_flat[take], 4).tolist(),
                   np.round(report.force_model_flat[take], 4).tolist()],
        "frame_err": np.round(frame_err, 4).tolist(),
        "frame_max": np.round(frame_max, 4).tolist(),
        "energy_error_meV_per_atom": np.round(report.energy_error_per_atom * 1000.0, 3).tolist(),
        "offsets_eV_per_atom": report.offsets,
    }


# ==========================================================================
# L1
# ==========================================================================
def rescore_l1(pack: ModelPack, ref: Reference) -> tuple[ResultSet, dict[str, Any]]:
    from .cache import pecs_from_npz
    from .l1_pes import (LatticeScan, l1_gate, lattice_scan_metrics, nve_metrics, pec_metrics,
                         probe_metrics, stability_metrics, timestep_metrics)
    from .md_runner import load_md

    level = pack.level("L1")
    settings = level.get("settings", {})
    ref_l1 = ref.l1()
    aimd = ref_l1["aimd_nve_reference"]
    results = ResultSet(
        "L1",
        model=pack.spec,
        probe_structure=f"{ref.reference_id}:{ref_l1['probe_structure']['file']} "
                        f"({ref_l1['probe_structure']['task']})",
        probe_protocol=ref_l1["probe_protocol"],
        aimd_nve_reference=aimd,
        quick=bool(settings.get("quick", False)),
    )
    plots: dict[str, Any] = {}

    if "pec" in level:
        pecs, _stale = pecs_from_npz(pack.npz(level["pec"]["file"]))
        results.extend(pec_metrics(pecs))
        pairs = sorted(pecs)
        step = 2  # 0.04 A on screen, as on the page; the full 0.02 A grid is in the pack
        plots["pec"] = {
            "r": np.round(pecs[pairs[0]].r[::step], 3).tolist(),
            "pairs": pairs,
            "e": [[None if not np.isfinite(v) else round(float(v), 4) for v in pecs[p].energy[::step]]
                  for p in pairs],
            "mono": [bool(pecs[p].monotonic_repulsive) for p in pairs],
        }

    if "probes" in level:
        probes = pack.json(level["probes"]["file"])
        results.context["measured_energy_noise_eV"] = probes["noise_eV"]
        results.extend(probe_metrics(probes))

    if "lattice_scan" in level:
        z = pack.npz(level["lattice_scan"]["file"])
        scan = LatticeScan(z["scales"], z["volumes"], z["energy_per_atom"], z["pressure_gpa"], z["min_distance"])
        results.extend(lattice_scan_metrics(scan))
        plots["scan"] = {"s": np.round(scan.scales, 3).tolist(),
                         "e": np.round(scan.energy_per_atom, 4).tolist(),
                         "d": np.round(scan.min_distance, 3).tolist()}

    if not settings.get("skip_md", False):
        if "stability_2000K" in level:
            hot = load_md(pack.path(level["stability_2000K"]["file"]))
            results.extend(stability_metrics(hot, temperature_k=float(settings.get("stability_temperature_K", 2000.0))))
        runs = {}
        for dt in settings.get("timesteps_fs", [3.0, 1.5]):
            member = f"nve_dt{float(dt):g}"
            if member in level:
                runs[float(dt)] = load_md(pack.path(level[member]["file"]))
        if runs:
            results.extend(timestep_metrics(runs, total_time_fs=float(settings.get("timestep_total_time_fs", 999.0))))
        if 3.0 in runs:
            results.extend(nve_metrics(runs[3.0], reference_drift=aimd["drift_meV_per_atom_per_ps"], timestep_fs=3.0))
            run = runs[3.0]
            e = run.total_energy / run.n_atoms * 1000.0
            plots["nve"] = {"t": np.round(run.time_fs / 1000.0, 4).tolist(),
                            "e": np.round(e - e[0], 5).tolist()}

    results.context["gate"] = l1_gate(results)
    return results, plots


# ==========================================================================
# L2
# ==========================================================================
def _eos_from_npz(z, label: str):
    from .l2_static import EOSCurve

    return EOSCurve(label=label, volumes=np.asarray(z["volumes"], dtype=float),
                    energies=np.asarray(z["energies"], dtype=float), n_atoms=int(z["n_atoms"]),
                    source=str(z["source"]) if "source" in z else "")


def rescore_l2(pack: ModelPack, ref: Reference) -> tuple[ResultSet, dict[str, Any]]:
    from .l2_static import (eos_ensemble_metrics, eos_metrics, eos_window, fit_eos,
                            formation_energy_metrics, phase_ladder_metrics, phase_ordering_metrics,
                            relaxation_record_metrics)

    level = pack.level("L2")
    relax = pack.json(level["relax"]["file"])
    records = relax["relaxations"]
    ref_l2 = ref.l2()
    tasks = ref_l2["tasks"]
    seed = int(relax.get("cubic_seed", 1))

    def dft(task: str) -> dict[str, Any]:
        if task not in tasks:
            raise PackError(f"reference pack has no DFT task {task}")
        return tasks[task]

    def need(label: str) -> dict[str, Any]:
        if label not in records:
            raise PackError(f"pack has no relaxation record {label!r}")
        return records[label]

    results = ResultSet(
        "L2",
        model=pack.spec,
        dft_protocol=ref_l2["dft_protocol"],
        fmax_eV_per_A=relax.get("fmax_eV_per_A"),
        eos_scales=relax.get("eos_scales"),
        eos_scale_kind=relax.get("eos_scale_kind", "linear"),
        note=L2_NOTE,
    )
    plots: dict[str, Any] = {}

    # same order as run_l2.py, so a rescored file lines up with the original
    hex_rec = need("hex_mp1224375")
    results.extend(relaxation_record_metrics(hex_rec, dft_reference=dft("L1_mp1224375_relax")))
    hex_static = dft("L1_mp1224375_static")
    cubic_label = f"cubic_s{seed:02d}"
    cubic_rec = need(cubic_label)
    results.extend(relaxation_record_metrics(cubic_rec, dft_reference=dft(f"H1_cubic_s{seed:02d}_relax")))
    cubic_static = dft(f"H1_cubic_s{seed:02d}_static")
    results.extend(phase_ordering_metrics(
        {"cubic": cubic_rec["energy_per_atom_eV"], "hexagonal": hex_rec["energy_per_atom_eV"]},
        dft_energies_per_atom={"cubic": cubic_static["energy_per_atom_eV"],
                               "hexagonal": hex_static["energy_per_atom_eV"]},
        reference_phase="hexagonal",
    ))

    ladder_model = {"mp1224375": hex_rec["energy_per_atom_eV"], "cubic": cubic_rec["energy_per_atom_eV"]}
    ladder_dft = {"mp1224375": hex_static["energy_per_atom_eV"], "cubic": cubic_static["energy_per_atom_eV"]}
    ladder_labels = {row[0]: row[3] for row in ref_l2["phase_ladder"]}
    compositions = {"mp1224375": hex_rec["composition"], "cubic": cubic_rec["composition"]}
    for key, _source, task, _label in ref_l2["phase_ladder"]:
        if key in ladder_model:
            continue
        rec = need(f"phase_{key}")
        results.extend(relaxation_record_metrics(rec, dft_reference=dft(task.replace("_static", "_relax"))))
        ladder_model[key] = rec["energy_per_atom_eV"]
        ladder_dft[key] = dft(task)["energy_per_atom_eV"]
        compositions[key] = rec["composition"]
    results.extend(phase_ladder_metrics(ladder_model, ladder_dft, labels=ladder_labels))

    model_mu, dft_mu = {}, {}
    for element in ref_l2["elements"]:
        tag = element.lower()
        rec = need(f"element_{element}")
        results.extend(relaxation_record_metrics(rec, dft_reference=dft(f"L2_{tag}_relax")))
        model_mu[element] = rec["energy_per_atom_eV"]
        dft_mu[element] = dft(f"L2_{tag}_static")["energy_per_atom_eV"]
    results.extend(formation_energy_metrics(ladder_model, ladder_dft, compositions, model_mu, dft_mu,
                                            labels=ladder_labels))

    eos_files = level.get("eos", {})
    windows: dict[str, Any] = {}
    if "hex" in eos_files and "hex" in ref_l2["eos"]:
        dft_hex = ref.eos_curve("hex")
        mlip_hex = _eos_from_npz(pack.npz(eos_files["hex"]), "hex_mlip")
        results.extend(eos_metrics("hex", fit_eos(mlip_hex), fit_eos(dft_hex)))
        windows["hex"] = eos_window(mlip_hex, dft_hex)
        plots["hex"] = _eos_plot(dft_hex, mlip_hex)
    per_seed = {}
    for s in range(1, 6):
        name, phase = f"cubic_s{s:02d}", f"cub{s:02d}"
        if name not in eos_files or phase not in ref_l2["eos"]:
            continue
        dft_curve = ref.eos_curve(phase)
        mlip_curve = _eos_from_npz(pack.npz(eos_files[name]), f"{name}_mlip")
        per_seed[f"s{s:02d}"] = (fit_eos(mlip_curve), fit_eos(dft_curve))
        windows[name] = eos_window(mlip_curve, dft_curve)
        if s == seed:
            plots["cubic"] = _eos_plot(dft_curve, mlip_curve)
    if per_seed:
        results.extend(eos_ensemble_metrics("cubic", per_seed))
    results.context["eos_window"] = windows
    if windows and not all(w["matched"] for w in windows.values()):
        results.context["eos_window_warning"] = (
            "MLIP and DFT EOS sample different V/V0 windows; V0/B0 errors include fit-range bias")
    return results, plots


def _eos_plot(dft, mlip) -> dict[str, Any]:
    """Workbench shape: V/atom and E - min(E) in meV/atom."""
    def one(c):
        v = c.volume_per_atom
        e = c.energy_per_atom
        e = (e - np.nanmin(e)) * 1000.0
        return {"v": np.round(v, 3).tolist(), "e": np.round(e, 4).tolist()}

    return {"dft": one(dft), "mlip": one(mlip)}


# ==========================================================================
# driver
# ==========================================================================
def rescore(pack: ModelPack, ref: Reference, *, levels: Iterable[str] = LEVELS) -> Rescored:
    """Verify, then rescore every requested level the pack has data for."""
    levels = [lv.upper() for lv in levels]
    verification = verify_model_pack(pack, ref)
    out = Rescored(pack=pack, reference=ref, verification=verification)
    warnings_by_level: dict[str, list] = {}
    for c in verification["checks"]:
        if not c["ok"] and c["severity"] == "warning":
            warnings_by_level.setdefault(c["level"], []).append(c)

    for lv in levels:
        if lv not in LEVELS:
            raise ValueError(f"cannot rescore {lv}: only {', '.join(LEVELS)} are rescored from packs")
        if not pack.level(lv):
            out.skipped[lv] = "the pack carries no data for this level"
            continue
        if lv in verification["blocked_levels"]:
            out.skipped[lv] = "verification failed: " + "; ".join(
                c["detail"] for c in verification["checks"]
                if not c["ok"] and c["severity"] == "error" and c["level"] in (lv, "-"))
            continue
        warns = warnings_by_level.get(lv, []) + warnings_by_level.get("-", [])
        if lv == "L0":
            sets, plots = rescore_l0(pack, ref)
            for stem, rs in sets.items():
                _stamp(rs, pack, ref, warns)
                out.results[stem] = rs
            out.plots["l0"] = plots
        elif lv == "L1":
            rs, plots = rescore_l1(pack, ref)
            _stamp(rs, pack, ref, warns)
            out.results["L1"] = rs
            out.plots["l1"] = plots
        else:
            rs, plots = rescore_l2(pack, ref)
            _stamp(rs, pack, ref, warns)
            out.results["L2"] = rs
            out.plots["l2"] = plots
    return out


# ==========================================================================
# comparison with the HPC original
# ==========================================================================
def _close(a: Any, b: Any, rtol: float, atol: float) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not (np.isfinite(a) and np.isfinite(b)):
            return (not np.isfinite(a)) and (not np.isfinite(b))
        return bool(np.isclose(a, b, rtol=rtol, atol=atol))
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return set(a) == set(b) and all(_close(a[k], b[k], rtol, atol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y, rtol, atol) for x, y in zip(a, b))
    return a == b


#: Metrics a backfilled pack cannot reproduce to full precision, with the
#: absolute tolerance that the rounding in the result file allows.
BACKFILL_TOLERANCE = {".cell_length_error": 5e-3}

#: Metrics that come out of an iterative fit (scipy ``curve_fit``): a different
#: scipy/BLAS build converges to the same minimum within ~1e-5 relative.
FIT_RTOL = {"L2.eos.": 1e-4}


def compare_results(rescored: ResultSet | Mapping[str, Any], original: Mapping[str, Any], *,
                    rtol: float = 1e-6, atol: float = 1e-9, backfilled: bool = False) -> dict[str, Any]:
    """Metric-by-metric agreement of value, status and grade."""
    new = rescored.to_dict() if isinstance(rescored, ResultSet) else rescored
    a = {r["metric_id"]: r for r in new["results"]}
    b = {r["metric_id"]: r for r in original["results"]}
    diffs = []
    for mid in sorted(set(a) & set(b)):
        x, y = a[mid], b[mid]
        tol = atol
        if backfilled:
            for suffix, t in BACKFILL_TOLERANCE.items():
                if mid.endswith(suffix):
                    tol = max(tol, t)
        r = max([rtol] + [t for prefix, t in FIT_RTOL.items() if mid.startswith(prefix)])
        problems = []
        if not _close(x["value"], y["value"], r, tol):
            problems.append("value")
        for key in ("status", "grade", "unit"):
            if x.get(key) != y.get(key):
                problems.append(key)
        if problems:
            diffs.append({"metric_id": mid, "fields": problems,
                          "rescored": {k: x.get(k) for k in ("value", "status", "grade")},
                          "original": {k: y.get(k) for k in ("value", "status", "grade")}})
    return {
        "n_compared": len(set(a) & set(b)),
        "n_agree": len(set(a) & set(b)) - len(diffs),
        "only_rescored": sorted(set(a) - set(b)),
        "only_original": sorted(set(b) - set(a)),
        "diffs": diffs,
        "identical": not diffs and set(a) == set(b),
    }


def compare_with_pack(out: Rescored) -> dict[str, Any]:
    """Compare every rescored file with the original the pack carries."""
    backfilled = bool(out.pack.manifest.get("provenance", {}).get("backfilled"))
    report = {}
    for stem, rs in out.results.items():
        original = out.pack.original(stem if stem in ("L1", "L2") else f"{stem}.json")
        if original is None:
            report[stem] = {"identical": None, "note": "no original in the pack"}
            continue
        report[stem] = compare_results(rs, original, backfilled=backfilled and stem == "L2")
    return report


# ==========================================================================
# Workbench view
# ==========================================================================
def workbench_model(out: Rescored) -> dict[str, Any]:
    """One entry of the Workbench ``D.models`` object, for the rescored levels.

    ``metrics`` rows are ``[metric_id, value, unit, status, grade, z_eff,
    uncertainty, tier, failure_reason]``, the layout the page reads.
    """
    rows = []
    for rs in out.results.values():
        for r in rs:
            v = r.value
            if isinstance(v, (dict, list, np.ndarray)):
                v = None
            elif isinstance(v, float) and not np.isfinite(v):
                v = None
            rows.append([r.metric_id, v, r.unit, r.status.value, r.grade.value, r.z_eff, r.uncertainty,
                         r.evidence_tier.value if r.evidence_tier else None, r.failure_reason])
    gate = out.results["L1"].context.get("gate") if "L1" in out.results else None
    entry: dict[str, Any] = {
        "label": out.pack.manifest.get("label", out.pack.model_key),
        "protocol": out.pack.manifest.get("model", {}).get("native_protocol"),
        "source": {"pack_id": out.pack.pack_id, "engine": RESCORE_VERSION},
        "metrics": rows,
    }
    if gate:
        entry["gate"] = {"pass": gate["pass"], "n": gate["n_checks"], "failed": gate["failed_metrics"],
                         "decision": gate["decision"]}
    if "l0" in out.plots and "audit" in out.plots["l0"]:
        a = out.plots["l0"]["audit"]
        entry["l0"] = {k: a[k] for k in ("parity", "frame_err", "frame_max")}
    if "l1" in out.plots:
        entry["l1"] = out.plots["l1"]
    if "l2" in out.plots:
        entry["l2"] = out.plots["l2"]
    return entry
