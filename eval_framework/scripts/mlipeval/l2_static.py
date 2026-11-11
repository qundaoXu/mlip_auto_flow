"""L2 -- static equilibrium properties.

Scope (``validation_methods_L0-L5.md`` L2):

1. relaxation of cubic and hexagonal GST225;
2. relative energy and phase ordering;
3. fixed-volume, ionic-relaxed EOS for each phase;
4. equilibrium volume and bulk modulus.

The DFT side comes from the completed phase-1 tasks (``H1_*`` cubic relax/static,
``H2_hex_*`` hexagonal EOS), which are tier-B evidence: same project protocol,
static DFT.  The 306-atom ``NSW=0`` volume sweep in ``GST225_ref`` is explicitly
*not* accepted here as an EOS -- :func:`load_dft_eos` only reads ionic-relaxed
fixed-volume tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from ase import Atoms

from .calculators import Model
from .io_vasp import outcar_is_complete, read_outcar_final
from .schema import EvidenceTier, MetricResult, Status


def _result(metric_id: str, value, unit: str, **kw) -> MetricResult:
    kw.setdefault("evidence_tier", EvidenceTier.B)
    return MetricResult(metric_id=metric_id, level="L2", value=value, unit=unit, **kw)


# --------------------------------------------------------------------------
# relaxation
# --------------------------------------------------------------------------
@dataclass
class RelaxResult:
    label: str
    initial: Atoms
    relaxed: Atoms
    energy: float
    energy_per_atom: float
    n_steps: int
    fmax: float
    max_force: float
    converged: bool
    volume_change_pct: float
    cell_lengths: np.ndarray
    cell_angles: np.ndarray

    def record(self) -> dict[str, object]:
        """Unrounded outcome of the relaxation: everything L2 scores it on.

        ``run_l2.py`` writes these to ``l2_relax.json`` so an evaluation pack can
        be rescored without the model; :func:`relaxation_record_metrics` reads
        them back.
        """
        symbols = self.relaxed.get_chemical_symbols()
        return {
            "label": self.label,
            "n_atoms": len(self.relaxed),
            "composition": {el: symbols.count(el) for el in sorted(set(symbols))},
            "energy_eV": self.energy,
            "energy_per_atom_eV": self.energy_per_atom,
            "volume_A3": float(self.relaxed.get_volume()),
            "volume_per_atom_A3": float(self.relaxed.get_volume() / len(self.relaxed)),
            "volume_change_pct": self.volume_change_pct,
            "cell_lengths_A": np.asarray(self.cell_lengths, dtype=float).tolist(),
            "cell_angles_deg": np.asarray(self.cell_angles, dtype=float).tolist(),
            "max_force_eV_per_A": self.max_force,
            "fmax": self.fmax,
            "n_steps": self.n_steps,
            "converged": self.converged,
        }

    def summary(self) -> dict[str, object]:
        return relax_summary(self.record())


def relax_summary(record: Mapping) -> dict[str, object]:
    """The display form of a relaxation record (what result files carry in ``extra``)."""
    return {
        "label": record["label"],
        "n_atoms": int(record["n_atoms"]),
        "energy_per_atom_eV": record["energy_per_atom_eV"],
        "volume_A3": record["volume_A3"],
        "volume_per_atom_A3": record["volume_per_atom_A3"],
        "volume_change_pct": record["volume_change_pct"],
        "cell_lengths_A": np.round(np.asarray(record["cell_lengths_A"], dtype=float), 4).tolist(),
        "cell_angles_deg": np.round(np.asarray(record["cell_angles_deg"], dtype=float), 3).tolist(),
        "max_force_eV_per_A": record["max_force_eV_per_A"],
        "n_steps": record["n_steps"],
        "converged": record["converged"],
    }


def relax_structure(
    model: Model,
    atoms: Atoms,
    *,
    label: str = "",
    relax_cell: bool = True,
    fmax: float = 0.01,
    max_steps: int = 400,
    optimizer: str = "FIRE",
    logfile: str | None = None,
) -> RelaxResult:
    """Relax ions (and optionally the cell) with the MLIP.

    ``FrechetCellFilter`` is used for variable-cell relaxation because it is the
    ASE filter with the correct cell gradient for large strains.
    """
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE, LBFGS

    work = atoms.copy()
    work.calc = model.calculator
    target = FrechetCellFilter(work) if relax_cell else work
    Optimizer = {"FIRE": FIRE, "LBFGS": LBFGS}[optimizer]
    opt = Optimizer(target, logfile=logfile or "/dev/null")
    opt.run(fmax=fmax, steps=max_steps)

    forces = np.asarray(work.get_forces())
    max_force = float(np.linalg.norm(forces, axis=1).max())
    v0, v1 = atoms.get_volume(), work.get_volume()
    return RelaxResult(
        label=label or atoms.get_chemical_formula(),
        initial=atoms,
        relaxed=work,
        energy=float(work.get_potential_energy()),
        energy_per_atom=float(work.get_potential_energy() / len(work)),
        n_steps=int(opt.get_number_of_steps()),
        fmax=fmax,
        max_force=max_force,
        converged=bool(max_force <= fmax),
        volume_change_pct=float(100.0 * (v1 - v0) / v0),
        cell_lengths=np.asarray(work.cell.lengths()),
        cell_angles=np.asarray(work.cell.angles()),
    )


def relaxation_metrics(
    relax: RelaxResult,
    *,
    dft_reference: Mapping[str, float] | None = None,
) -> list[MetricResult]:
    """Compare relaxed geometry against the DFT relaxation of the same seed."""
    return relaxation_record_metrics(relax.record(), dft_reference=dft_reference)


def relaxation_record_metrics(
    record: Mapping,
    *,
    dft_reference: Mapping[str, float] | None = None,
) -> list[MetricResult]:
    """:func:`relaxation_metrics` on a stored :meth:`RelaxResult.record`."""
    label = record["label"]
    converged = bool(record["converged"])
    results = [
        _result(
            f"L2.relax.{label}.energy_per_atom",
            record["energy_per_atom_eV"],
            "eV/atom",
            status=Status.OK if converged else Status.FAILED,
            failure_reason=None if converged else f"fmax {record['max_force_eV_per_A']:.4f} > {record['fmax']}",
            evidence_tier=EvidenceTier.S,
            extra=relax_summary(record),
        ),
    ]
    v_model = float(record["volume_per_atom_A3"])
    cell_lengths = np.asarray(record["cell_lengths_A"], dtype=float)
    if dft_reference and "volume_per_atom_A3" in dft_reference:
        v_ref = float(dft_reference["volume_per_atom_A3"])
        results.append(
            _result(
                f"L2.relax.{label}.volume_error",
                float(100.0 * (v_model - v_ref) / v_ref),
                "%",
                status=Status.PROVISIONAL,
                extra={"model_A3_per_atom": v_model, "dft_A3_per_atom": v_ref},
            )
        )
    if dft_reference and "cell_lengths_A" in dft_reference:
        ref_len = np.asarray(dft_reference["cell_lengths_A"], dtype=float)
        err = 100.0 * (cell_lengths - ref_len) / ref_len
        results.append(
            _result(
                f"L2.relax.{label}.cell_length_error",
                float(np.abs(err).max()),
                "%",
                status=Status.PROVISIONAL,
                extra={"per_axis_pct": np.round(err, 3).tolist(), "dft_A": ref_len.tolist()},
            )
        )
    return results


# --------------------------------------------------------------------------
# equation of state
# --------------------------------------------------------------------------
@dataclass
class EOSCurve:
    label: str
    volumes: np.ndarray          # A^3 (whole cell)
    energies: np.ndarray         # eV (whole cell)
    n_atoms: int
    source: str = ""
    relaxed_ions: bool = True

    @property
    def volume_per_atom(self) -> np.ndarray:
        return self.volumes / self.n_atoms

    @property
    def energy_per_atom(self) -> np.ndarray:
        return self.energies / self.n_atoms


@dataclass
class EOSFit:
    v0_per_atom: float
    e0_per_atom: float
    b0_gpa: float
    bp: float
    rmse_mev_per_atom: float
    n_points: int
    model: str = "birch-murnaghan"

    def to_dict(self) -> dict[str, float | str | int]:
        return {
            "V0_A3_per_atom": self.v0_per_atom,
            "E0_eV_per_atom": self.e0_per_atom,
            "B0_GPa": self.b0_gpa,
            "Bp": self.bp,
            "fit_rmse_meV_per_atom": self.rmse_mev_per_atom,
            "n_points": self.n_points,
            "model": self.model,
        }


def birch_murnaghan(volume: np.ndarray, e0: float, v0: float, b0: float, bp: float) -> np.ndarray:
    """Third-order Birch-Murnaghan E(V); ``b0`` in eV/A^3."""
    eta = (v0 / volume) ** (2.0 / 3.0)
    return e0 + 9.0 * v0 * b0 / 16.0 * ((eta - 1.0) ** 3 * bp + (eta - 1.0) ** 2 * (6.0 - 4.0 * eta))


def fit_eos(curve: EOSCurve) -> EOSFit:
    """Third-order Birch-Murnaghan fit on a per-atom basis."""
    from ase.units import GPa
    from scipy.optimize import curve_fit

    v = curve.volume_per_atom
    e = curve.energy_per_atom
    ok = np.isfinite(v) & np.isfinite(e)
    v, e = v[ok], e[ok]
    if v.size < 4:
        return EOSFit(np.nan, np.nan, np.nan, np.nan, np.nan, int(v.size))

    # parabola in V gives a robust starting point
    c = np.polyfit(v, e, 2)
    v0_guess = -c[1] / (2.0 * c[0])
    b0_guess = 2.0 * c[0] * v0_guess
    p0 = [float(np.min(e)), float(v0_guess), float(max(b0_guess, 1e-3)), 4.0]
    try:
        popt, _ = curve_fit(birch_murnaghan, v, e, p0=p0, maxfev=20000)
    except Exception:
        return EOSFit(np.nan, np.nan, np.nan, np.nan, np.nan, int(v.size))
    e0, v0, b0, bp = popt
    residual = e - birch_murnaghan(v, *popt)
    return EOSFit(
        v0_per_atom=float(v0),
        e0_per_atom=float(e0),
        b0_gpa=float(b0 / GPa),
        bp=float(bp),
        rmse_mev_per_atom=float(np.sqrt(np.mean(residual**2)) * 1000.0),
        n_points=int(v.size),
    )


def mlip_eos(
    model: Model,
    atoms: Atoms,
    *,
    label: str,
    scales: Sequence[float] = (0.92, 0.95, 0.98, 1.00, 1.02, 1.05, 1.08),
    relax_ions: bool = True,
    fmax: float = 0.01,
    max_steps: int = 200,
    scale_kind: str = "volume",
) -> EOSCurve:
    """Fixed-volume, ionic-relaxed EOS -- the same construction as the H2 tasks.

    ``scales`` are *volume* scale factors by default, matching ``H2_*_v092..v108``
    (their cells are the reference cell times ``s**(1/3)``: V/V0 = 0.92..1.08).
    ``scale_kind="linear"`` reproduces the scan used before 2026-11-11, which
    applied the factors to the cell vectors and so spanned V/V0 = 0.78..1.26 --
    a Birch-Murnaghan fit over three times the DFT window, which biases B0 and
    makes the V0/B0 errors partly a statement about the fit range.
    """
    if scale_kind not in ("volume", "linear"):
        raise ValueError(f"scale_kind must be 'volume' or 'linear', not {scale_kind!r}")
    volumes, energies = [], []
    for s in scales:
        scaled = atoms.copy()
        factor = s ** (1.0 / 3.0) if scale_kind == "volume" else s
        scaled.set_cell(atoms.cell.array * factor, scale_atoms=True)
        if relax_ions:
            relaxed = relax_structure(
                model, scaled, label=f"{label}_v{s:.3f}", relax_cell=False, fmax=fmax, max_steps=max_steps
            )
            volumes.append(relaxed.relaxed.get_volume())
            energies.append(relaxed.energy)
        else:
            scaled.calc = model.calculator
            volumes.append(scaled.get_volume())
            energies.append(float(scaled.get_potential_energy()))
    return EOSCurve(
        label=label,
        volumes=np.array(volumes),
        energies=np.array(energies),
        n_atoms=len(atoms),
        source=(f"{model.spec.name} fixed-volume {'ionic-relaxed' if relax_ions else 'unrelaxed'} scan, "
                f"{scale_kind} scale factors {list(scales)}"),
        relaxed_ions=relax_ions,
    )


def load_dft_eos(
    task_dirs: Iterable[Path | str],
    *,
    label: str,
    require_ionic_relaxation: bool = True,
) -> EOSCurve:
    """Read completed fixed-volume EOS tasks into an :class:`EOSCurve`.

    Tasks whose INCAR has ``NSW = 0`` are rejected when
    ``require_ionic_relaxation`` is set, because an unrelaxed volume scan is not
    an EOS (``validation_methods_L0-L5.md`` L2).
    """
    from .io_vasp import read_incar, task_output_dir

    volumes, energies, used, skipped = [], [], [], []
    n_atoms = 0
    for task in sorted(Path(d) for d in task_dirs):
        directory = task_output_dir(task)
        outcar = directory / "OUTCAR"
        if not outcar_is_complete(outcar):
            skipped.append(f"{task.name}: incomplete")
            continue
        tags = read_incar(directory / "INCAR")
        nsw = int(float(tags.get("NSW", "0")))
        if require_ionic_relaxation and nsw <= 0:
            skipped.append(f"{task.name}: NSW={nsw}, not an ionic relaxation")
            continue
        final = read_outcar_final(outcar)
        volumes.append(final["volume"])
        energies.append(final["energy"])
        n_atoms = int(final["n_atoms"])
        used.append(task.name)

    curve = EOSCurve(
        label=label,
        volumes=np.array(volumes),
        energies=np.array(energies),
        n_atoms=n_atoms or 1,
        source=f"phase-1 DFT tasks: {', '.join(used) if used else 'none'}",
        relaxed_ions=require_ionic_relaxation,
    )
    curve.skipped = skipped  # type: ignore[attr-defined]
    return curve


def eos_window(mlip: EOSCurve, dft: EOSCurve, *, rel_tol: float = 0.25) -> dict[str, object]:
    """Do the MLIP and DFT curves sample the same relative volume window?

    A Birch-Murnaghan B0 depends on the range it is fitted over, so V0/B0 errors
    are only like-for-like when both curves span the same V/V0.  Spans are
    compared relative to each curve's own median volume, so a model whose V0 is
    off by a percent still matches.
    """
    def span(curve: EOSCurve) -> tuple[float, float, float]:
        v = np.asarray(curve.volume_per_atom, dtype=float)
        v = v[np.isfinite(v)]
        if v.size < 2:
            return float("nan"), float("nan"), float("nan")
        mid = float(np.median(v))
        return float(v.min() / mid), float(v.max() / mid), float((v.max() - v.min()) / mid)

    m_lo, m_hi, m_span = span(mlip)
    d_lo, d_hi, d_span = span(dft)
    matched = bool(np.isfinite(m_span) and np.isfinite(d_span) and d_span > 0
                   and abs(m_span - d_span) / d_span <= rel_tol)
    return {
        "matched": matched,
        "mlip_V_over_Vmid": [m_lo, m_hi],
        "dft_V_over_Vmid": [d_lo, d_hi],
        "span_ratio_mlip_to_dft": (m_span / d_span) if d_span and np.isfinite(d_span) else None,
        "note": None if matched else (
            "the MLIP EOS spans a different V/V0 window from the DFT tasks; the V0/B0 "
            "errors then mix model error with fit-range bias (B0 most of all)"),
    }


def eos_metrics(
    label: str,
    mlip_fit: EOSFit,
    dft_fit: EOSFit | None = None,
    *,
    n_independent_references: int = 1,
) -> list[MetricResult]:
    results = [
        _result(
            f"L2.eos.{label}.V0",
            mlip_fit.v0_per_atom,
            "A^3/atom",
            status=Status.PROVISIONAL,
            evidence_tier=EvidenceTier.S if dft_fit is None else EvidenceTier.B,
            extra=mlip_fit.to_dict(),
        ),
        _result(
            f"L2.eos.{label}.B0",
            mlip_fit.b0_gpa,
            "GPa",
            status=Status.PROVISIONAL,
            evidence_tier=EvidenceTier.S if dft_fit is None else EvidenceTier.B,
            extra=mlip_fit.to_dict(),
        ),
    ]
    if dft_fit is not None and np.isfinite(dft_fit.v0_per_atom):
        dv = 100.0 * (mlip_fit.v0_per_atom - dft_fit.v0_per_atom) / dft_fit.v0_per_atom
        db = 100.0 * (mlip_fit.b0_gpa - dft_fit.b0_gpa) / dft_fit.b0_gpa
        results.append(
            _result(
                f"L2.eos.{label}.V0_error",
                float(dv),
                "%",
                status=Status.PROVISIONAL,
                extra={"dft": dft_fit.to_dict(), "mlip": mlip_fit.to_dict()},
            )
        )
        results.append(
            _result(
                f"L2.eos.{label}.B0_error",
                float(db),
                "%",
                status=Status.PROVISIONAL,
                extra={"dft": dft_fit.to_dict(), "mlip": mlip_fit.to_dict()},
            )
        )
    return results


#: Why the cubic seed spread is *not* a ``sigma_ref``.  The five ``H2_cub0*``
#: EOS curves come from five different vacancy orderings of the same 194-atom
#: rocksalt cell, not five repeats of one measurement, so their scatter measures
#: how much the cubic phase's own EOS depends on how the vacancies sit.  That is
#: a property of the phase, not the noise on a reference value, and it therefore
#: cannot enter ``sigma_eff = max(sigma_ref, d_protocol/2)``.  What it *does*
#: license is a statement about the model's error across configurations, which
#: is what the paired per-seed errors below report.
SEED_SPREAD_NOTE = (
    "five different vacancy orderings of the 194-atom cubic cell, not replicas: "
    "this spread is configurational, not a sigma_ref, and does not license a grade"
)


def eos_ensemble_metrics(
    label: str,
    per_seed: Mapping[str, tuple[EOSFit, EOSFit]],
) -> list[MetricResult]:
    """Paired MLIP-vs-DFT EOS fits over several structural seeds, as one entry.

    ``per_seed`` maps a seed key to ``(mlip_fit, dft_fit)``.  Each seed is a
    genuine paired comparison -- same vacancy ordering on both sides -- so the
    spread of the *errors* separates the model's systematic offset from the part
    of its error that depends on which configuration it was handed.

    This deliberately emits **one** ``L2.eos.{label}.V0_error`` (and ``B0``)
    holding the seed-mean, rather than one metric per seed.  The ranking's EOS
    indicators aggregate with ``_mean_abs`` over ``^L2\\.eos\\..*\\.V0_error$``,
    so five per-seed entries would silently give the cubic phase five times the
    weight of the hexagonal one in the cross-model score.  The per-seed values
    are kept in ``extra.per_seed`` and in the ``seed_ensemble`` metrics below,
    which are named so that they do not match that pattern.

    The reference spread is reported as its own metric and carries
    :data:`SEED_SPREAD_NOTE`, so it can never be mistaken for the statistical
    ``sigma_ref`` the grading rule asks for.
    """
    usable = {
        seed: (m, d)
        for seed, (m, d) in per_seed.items()
        if np.isfinite(m.v0_per_atom) and np.isfinite(d.v0_per_atom) and np.isfinite(d.b0_gpa)
    }
    if not usable:
        return []

    seeds = sorted(usable)
    dft_v0 = np.array([usable[s][1].v0_per_atom for s in seeds])
    dft_b0 = np.array([usable[s][1].b0_gpa for s in seeds])
    mlip_v0 = np.array([usable[s][0].v0_per_atom for s in seeds])
    mlip_b0 = np.array([usable[s][0].b0_gpa for s in seeds])
    v0_err = 100.0 * (mlip_v0 - dft_v0) / dft_v0
    b0_err = 100.0 * (mlip_b0 - dft_b0) / dft_b0

    def _sd(a: np.ndarray) -> float:
        return float(np.std(a, ddof=1)) if a.size > 1 else float("nan")

    common = {
        "seeds": seeds,
        "n_seeds": len(seeds),
        "note": SEED_SPREAD_NOTE,
    }
    per_seed_detail = {
        s: {
            "dft_V0_A3_per_atom": float(usable[s][1].v0_per_atom),
            "mlip_V0_A3_per_atom": float(usable[s][0].v0_per_atom),
            "dft_B0_GPa": float(usable[s][1].b0_gpa),
            "mlip_B0_GPa": float(usable[s][0].b0_gpa),
            "V0_error_pct": float(100.0 * (usable[s][0].v0_per_atom - usable[s][1].v0_per_atom) / usable[s][1].v0_per_atom),
            "B0_error_pct": float(100.0 * (usable[s][0].b0_gpa - usable[s][1].b0_gpa) / usable[s][1].b0_gpa),
        }
        for s in seeds
    }

    results = [
        # the single scored entry per label: seed-mean MLIP value and seed-mean
        # paired error, each carrying its across-seed spread as the uncertainty
        _result(
            f"L2.eos.{label}.V0",
            float(mlip_v0.mean()),
            "A^3/atom",
            status=Status.PROVISIONAL,
            evidence_tier=EvidenceTier.B,
            uncertainty=_sd(mlip_v0),
            extra={**common, "per_seed": per_seed_detail,
                   "value_meaning": "mean over seeds",
                   "uncertainty_meaning": "sd of the MLIP fit across vacancy orderings"},
        ),
        _result(
            f"L2.eos.{label}.B0",
            float(mlip_b0.mean()),
            "GPa",
            status=Status.PROVISIONAL,
            evidence_tier=EvidenceTier.B,
            uncertainty=_sd(mlip_b0),
            extra={**common, "per_seed": per_seed_detail,
                   "value_meaning": "mean over seeds",
                   "uncertainty_meaning": "sd of the MLIP fit across vacancy orderings"},
        ),
        _result(
            f"L2.eos.{label}.V0_error",
            float(v0_err.mean()),
            "%",
            status=Status.PROVISIONAL,
            evidence_tier=EvidenceTier.B,
            uncertainty=_sd(v0_err),
            extra={**common, "per_seed": per_seed_detail,
                   "value_meaning": "mean paired error over seeds",
                   "uncertainty_meaning": "sd of the paired error across vacancy orderings"},
        ),
        _result(
            f"L2.eos.{label}.B0_error",
            float(b0_err.mean()),
            "%",
            status=Status.PROVISIONAL,
            evidence_tier=EvidenceTier.B,
            uncertainty=_sd(b0_err),
            extra={**common, "per_seed": per_seed_detail,
                   "value_meaning": "mean paired error over seeds",
                   "uncertainty_meaning": "sd of the paired error across vacancy orderings"},
        ),
        # what the reference itself does across vacancy orderings
        _result(
            f"L2.eos.{label}.seed_ensemble.dft_V0_spread",
            _sd(dft_v0),
            "A^3/atom",
            status=Status.PROVISIONAL,
            extra={**common, "dft_V0_mean_A3_per_atom": float(dft_v0.mean()),
                   "relative_pct": float(100.0 * _sd(dft_v0) / dft_v0.mean())},
        ),
        _result(
            f"L2.eos.{label}.seed_ensemble.dft_B0_spread",
            _sd(dft_b0),
            "GPa",
            status=Status.PROVISIONAL,
            extra={**common, "dft_B0_mean_GPa": float(dft_b0.mean()),
                   "relative_pct": float(100.0 * _sd(dft_b0) / dft_b0.mean())},
        ),
    ]

    # Is the model's error actually resolvable against how much the reference
    # moves between vacancy orderings?  A mean error smaller than the reference's
    # own configurational spread is not a measurement of the model.
    for name, err, ref, unit in (
        ("V0", v0_err, 100.0 * _sd(dft_v0) / dft_v0.mean(), "%"),
        ("B0", b0_err, 100.0 * _sd(dft_b0) / dft_b0.mean(), "%"),
    ):
        resolvable = bool(np.isfinite(ref) and abs(err.mean()) > ref)
        results.append(
            _result(
                f"L2.eos.{label}.seed_ensemble.{name}_error_vs_seed_spread",
                float(abs(err.mean()) / ref) if np.isfinite(ref) and ref > 0 else float("nan"),
                "ratio",
                status=Status.PROVISIONAL,
                extra={
                    **common,
                    "mean_abs_error_pct": float(abs(err.mean())),
                    "dft_seed_spread_pct": float(ref),
                    "error_exceeds_seed_spread": resolvable,
                    "interpretation": (
                        "ratio > 1: the model's mean error is larger than how much the DFT "
                        "reference itself moves between vacancy orderings, so it is a "
                        "statement about the model.  ratio < 1: it is not."
                    ),
                },
            )
        )
    return results


# --------------------------------------------------------------------------
# phase ordering
# --------------------------------------------------------------------------
def phase_ordering_metrics(
    energies_per_atom: Mapping[str, float],
    *,
    dft_energies_per_atom: Mapping[str, float] | None = None,
    reference_phase: str | None = None,
) -> list[MetricResult]:
    """Relative phase energies and whether the model reproduces the DFT ordering."""
    phases = list(energies_per_atom)
    if not phases:
        return []
    base = reference_phase or min(phases, key=lambda p: energies_per_atom[p])
    results: list[MetricResult] = []
    for phase in phases:
        if phase == base:
            continue
        delta = (energies_per_atom[phase] - energies_per_atom[base]) * 1000.0
        extra: dict[str, object] = {"reference_phase": base, "note": "same composition, same protocol"}
        status = Status.PROVISIONAL
        if dft_energies_per_atom and phase in dft_energies_per_atom and base in dft_energies_per_atom:
            dft_delta = (dft_energies_per_atom[phase] - dft_energies_per_atom[base]) * 1000.0
            extra.update({"dft_meV_per_atom": dft_delta, "error_meV_per_atom": delta - dft_delta})
            same_sign = np.sign(delta) == np.sign(dft_delta)
            extra["ordering_reproduced"] = bool(same_sign)
            status = Status.PROVISIONAL if same_sign else Status.FAILED
        results.append(
            _result(
                f"L2.phase_order.{phase}_minus_{base}",
                float(delta),
                "meV/atom",
                status=status,
                failure_reason=None if status is not Status.FAILED else "phase ordering inverted vs DFT",
                extra=extra,
            )
        )
    return results

#: The four crystalline GST225 references, ordered by their DFT energy.  Two of
#: them share space group P-3m1: they are different Te-Ge-Te stacking sequences
#: of the trigonal phase, not different space groups, and the 15 meV/atom between
#: them is the sharpest discriminator in this ladder -- telling GST stacking
#: sequences apart is exactly what a crystallisation study needs.
PHASE_LADDER: tuple[tuple[str, str, str, str], ...] = (
    # key,        structure source,  DFT static task,          label
    ("mp1224375", "mp-1224375",      "L1_mp1224375_static",    "trigonal P-3m1 (mp-1224375)"),
    ("mp3534",    "mp-3534",         "L1_mp3534_static",       "trigonal P-3m1 (mp-3534, ICSD)"),
    ("mp1224407", "mp-1224407",      "L1_mp1224407_static",    "monoclinic Cm (mp-1224407)"),
    ("cubic",     "cubic_seed",      "H1_cubic_s{seed:02d}_static", "cubic rocksalt+vacancies"),
)


def formation_energies(
    energies_per_atom: Mapping[str, float],
    compositions: Mapping[str, Mapping[str, float]],
    chemical_potentials: Mapping[str, float],
) -> dict[str, float]:
    """``dH_f = E/atom - sum_X x_X mu_X`` in eV/atom, ``x_X`` the atomic fractions.

    ``mu_X`` is the energy per atom of the element in its ground state under the
    *same* energy model, so each side is referenced to its own elements: a
    per-element offset between the model's native protocol and this project's
    PBE cancels exactly, which is what makes the result comparable across
    models trained on different DFT settings.
    """
    out = {}
    for key, e in energies_per_atom.items():
        comp = compositions[key]
        total = sum(comp.values())
        out[key] = float(e - sum(n / total * chemical_potentials[el] for el, n in comp.items()))
    return out


def formation_energy_metrics(
    model_energies: Mapping[str, float],
    dft_energies: Mapping[str, float],
    compositions: Mapping[str, Mapping[str, float]],
    model_mu: Mapping[str, float],
    dft_mu: Mapping[str, float],
    *,
    hexagonal: str = "mp1224375",
    cubic: str = "cubic",
    labels: Mapping[str, str] | None = None,
) -> list[MetricResult]:
    """Formation energies against the L2 element references, and the corrected cubic-hexagonal gap.

    This is the comparison ``TASKS.md`` section 7 prescribes for hex vs cubic:
    the cubic seeds are Ge43Sb43Te108, not Ge2Sb2Te5, so a bare per-atom energy
    difference mixes the 0.2 Ge + 0.2 Sb excess into the phase gap.  Referencing
    both phases to the elements removes it; :func:`phase_ladder_metrics` keeps
    the bare same-cell comparison beside it.
    """
    labels = labels or {}
    keys = [k for k in model_energies if k in dft_energies]
    model_h = formation_energies({k: model_energies[k] for k in keys}, compositions, model_mu)
    dft_h = formation_energies({k: dft_energies[k] for k in keys}, compositions, dft_mu)
    reference = {"mu_dft_eV_per_atom": dict(dft_mu), "mu_model_eV_per_atom": dict(model_mu),
                 "elements": "L2_{ge,sb,te}_static (DFT) / MLIP-relaxed element cells (model)"}

    results: list[MetricResult] = []
    for key in sorted(keys, key=lambda k: dft_h[k]):
        results.append(_result(
            f"L2.formation_energy.dE_error.{key}",
            float((model_h[key] - dft_h[key]) * 1000.0),
            "meV/atom",
            status=Status.PROVISIONAL,
            extra={"phase": labels.get(key, key), "composition": dict(compositions[key]),
                   "model_meV_per_atom": model_h[key] * 1000.0,
                   "dft_meV_per_atom": dft_h[key] * 1000.0, **reference},
        ))
    if hexagonal in keys and cubic in keys:
        model_gap = (model_h[cubic] - model_h[hexagonal]) * 1000.0
        dft_gap = (dft_h[cubic] - dft_h[hexagonal]) * 1000.0
        same_sign = bool(np.sign(model_gap) == np.sign(dft_gap))
        results.append(_result(
            "L2.formation_energy.cubic_minus_hexagonal",
            float(model_gap),
            "meV/atom",
            status=Status.PROVISIONAL if same_sign else Status.FAILED,
            failure_reason=None if same_sign else "composition-corrected phase ordering inverted vs DFT",
            extra={"dft_meV_per_atom": dft_gap, "error_meV_per_atom": model_gap - dft_gap,
                   "ordering_reproduced": same_sign, "reference_phase": hexagonal,
                   "note": "both phases referenced to the element ground states (composition-corrected)",
                   **reference},
        ))
    return results


def phase_ladder_metrics(
    energies_per_atom: Mapping[str, float],
    dft_energies_per_atom: Mapping[str, float],
    *,
    labels: Mapping[str, str] | None = None,
) -> list[MetricResult]:
    """The whole energy ladder, not just its two ends.

    :func:`phase_ordering_metrics` compares every phase against one base, so with
    the DFT-lowest phase as base its sign test is satisfied by construction for
    all of them -- it only ever really tested one pair.  This tests **every**
    pair, which is what makes a four-rung ladder stronger than a binary: a model
    can put the extremes in the right order and still scramble the middle.

    Composition note: the cubic cell is Ge43Sb43Te108, i.e. Te/Ge = 2.512 where
    exact Ge2Sb2Te5 wants 2.500.  Model and DFT see the identical cell, so the
    offset is common-mode and the *errors* below are unaffected, but the absolute
    cubic rung must not be quoted as a phase enthalpy.
    """
    keys = [k for k in energies_per_atom if k in dft_energies_per_atom]
    if len(keys) < 2:
        return []
    labels = labels or {}
    base = min(keys, key=lambda k: dft_energies_per_atom[k])

    model_d = {k: (energies_per_atom[k] - energies_per_atom[base]) * 1000.0 for k in keys}
    dft_d = {k: (dft_energies_per_atom[k] - dft_energies_per_atom[base]) * 1000.0 for k in keys}

    results: list[MetricResult] = []

    # --- per-rung error ---------------------------------------------------
    for key in sorted(keys, key=lambda k: dft_d[k]):
        if key == base:
            continue
        err = model_d[key] - dft_d[key]
        results.append(_result(
            f"L2.phase_ladder.dE_error.{key}",
            float(err),
            "meV/atom",
            status=Status.PROVISIONAL,
            extra={
                "reference_phase": base,
                "phase": labels.get(key, key),
                "model_meV_per_atom": model_d[key],
                "dft_meV_per_atom": dft_d[key],
                "note": (
                    "same protocol, same cell; the mp3534 rung is the two P-3m1 "
                    "stackings and is the sharpest test in the ladder"
                    if key == "mp3534" else "same protocol, same cell"
                ),
            },
        ))

    # --- pairwise ordering over all C(n,2) pairs ---------------------------
    pairs = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:]]
    inverted = [
        (a, b) for a, b in pairs
        # a strict inversion: DFT separates the pair and the model gets the sign
        # backwards.  A DFT gap of exactly 0 cannot be ordered either way.
        if dft_d[a] != dft_d[b]
        and np.sign(model_d[a] - model_d[b]) != np.sign(dft_d[a] - dft_d[b])
    ]
    n_orderable = sum(1 for a, b in pairs if dft_d[a] != dft_d[b])
    results.append(_result(
        "L2.phase_ladder.inversions",
        float(len(inverted)),
        "count",
        status=Status.OK if not inverted else Status.FAILED,
        failure_reason=None if not inverted else "; ".join(
            f"{labels.get(a, a)} vs {labels.get(b, b)} ordered backwards" for a, b in inverted
        ),
        extra={
            "n_pairs": n_orderable,
            "inverted_pairs": [[a, b] for a, b in inverted],
            "dft_ladder_meV_per_atom": {k: dft_d[k] for k in keys},
            "model_ladder_meV_per_atom": {k: model_d[k] for k in keys},
            "reference_phase": base,
        },
    ))

    # Kendall tau on the same pair set: a single number for "how scrambled",
    # where the inversion count alone does not say out of how many.
    tau = 1.0 - 2.0 * len(inverted) / n_orderable if n_orderable else float("nan")
    results.append(_result(
        "L2.phase_ladder.kendall_tau",
        float(tau),
        "-",
        status=Status.OK,
        extra={"note": "1 = ladder reproduced exactly, -1 = fully reversed",
               "n_pairs": n_orderable},
    ))

    span = max(dft_d.values()) - min(dft_d.values())
    worst = max((abs(model_d[k] - dft_d[k]) for k in keys if k != base), default=float("nan"))
    results.append(_result(
        "L2.phase_ladder.max_dE_error",
        float(worst),
        "meV/atom",
        status=Status.PROVISIONAL,
        extra={"dft_ladder_span_meV_per_atom": span,
               "note": "worst rung error; compare against the span of the DFT ladder"},
    ))
    return results
