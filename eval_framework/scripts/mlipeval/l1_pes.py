"""L1 -- potential-energy-surface sanity and trajectory stability.

Reference-free (evidence tier S) checks that must pass before any expensive
L4/L5 task is worth running (``validation_methods_L0-L5.md`` L1):

1. binary potential-energy curves for all species pairs;
2. PEC continuity, short-range repulsion and energy/force derivative agreement;
3. translation / rotation invariance, net force, analytic-vs-numeric stress;
4. unrelaxed isotropic lattice scan over 0.85-1.15;
5. high-temperature stability and the minimum-interatomic-distance distribution;
6. timestep sensitivity (3.0 fs vs 1.5 fs);
7. NVE energy drift and centre-of-mass drift.

Every check returns a :class:`~mlipeval.schema.MetricResult` whose ``Status`` is
``FAILED`` when the model is pathological; :func:`l1_gate` turns the collection
into the go/no-go decision for L4 and L5.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Mapping, Sequence

import numpy as np
from ase import Atoms

from .calculators import Model
from .md_runner import MDResult, MDSettings, run_md
from .neighbors import pair_key
from .schema import EvidenceTier, MetricResult, ResultSet, Status

# --- thresholds -----------------------------------------------------------
#: A model is pathological below these, not merely inaccurate.
THRESHOLDS = {
    "pec_min_repulsion_r": 1.6,      # A: E(r) must still be rising as r shrinks below this
    "pec_force_mismatch_rel": 0.05,  # analytic vs finite-difference force along the bond
    "pec_convergence_ratio": 2.5,    # h -> h/2 must shrink the derivative residual ~4x
    "scan_smoothness_ratio": 25.0,   # max/median |second difference| of E(scale)
    "force_consistency_rel": 5e-3,   # |F_analytic - F_numeric| / max|F|
    "translation_energy_tol": 1e-6,  # eV
    "rotation_energy_tol": 1e-5,     # eV
    "net_force_tol": 1e-6,           # eV/A per atom
    "stress_consistency_rel": 5e-2,  # relative to max |stress|
    "nve_drift_tol": 5.0,            # meV/atom/ps
    "com_drift_tol": 1e-3,           # A
    "min_distance_floor": 1.6,       # A: closest approach seen in a stable GST melt
}

#: Multiple of the *measured* energy noise below which a residual is numerical
#: rather than physical.  The absolute thresholds above were written for a
#: float64 model; a float32 network evaluating a 306-atom cell carries an energy
#: noise floor of order 1e-4 eV, so a 1 ueV translation-invariance tolerance
#: would report its rounding error as a symmetry violation.  Every check below
#: therefore compares against ``max(absolute threshold, NOISE_MARGIN * measured
#: noise propagated to that quantity)`` and records both in ``extra``.
NOISE_MARGIN = 5.0


def energy_noise(model: Model, atoms: Atoms, *, n_probes: int = 4, seed: int = 0) -> float:
    """Measured energy noise floor, in eV, for this model on this structure.

    Three transformations that leave the energy *exactly* invariant are applied;
    whatever spread they produce is pure numerics -- dtype, non-deterministic
    scatter, neighbour-list and summation order.  Measuring it is more honest
    than assuming a dtype-based bound, because it also catches non-determinism
    that float64 alone would not explain.

    All three are needed.  A rigid translation of a periodic cell is invariant,
    but a model that builds its graph from fractional coordinates (CHGNet) maps
    it to a bit-identical graph and reports exactly zero noise -- while its true
    float32 floor is ~1e-4 eV on this cell.  Permuting the atom order changes
    the summation order without changing the physics and does expose it.

    If every probe returns exactly zero the measurement is discarded and a
    dtype-based bound is used instead, because "no noise at all" is a statement
    about the probe, not about the model.
    """
    rng = np.random.default_rng(seed)
    e0 = model.energy(atoms)
    spread = 0.0
    for _ in range(n_probes):
        shifted = atoms.copy()
        shifted.positions += rng.uniform(-3.0, 3.0, size=3)
        spread = max(spread, abs(model.energy(shifted) - e0))

        order = rng.permutation(len(atoms))
        permuted = atoms[order]
        spread = max(spread, abs(model.energy(permuted) - e0))

        both = permuted.copy()
        both.positions += rng.uniform(-3.0, 3.0, size=3)
        spread = max(spread, abs(model.energy(both) - e0))

    if spread == 0.0:
        eps = {"float32": np.finfo(np.float32).eps, "float64": np.finfo(np.float64).eps}
        spread = float(eps.get(model.spec.dtype, np.finfo(np.float32).eps) * abs(e0))
    return float(spread)


def _result(metric_id: str, value, unit: str, **kw) -> MetricResult:
    kw.setdefault("evidence_tier", EvidenceTier.S)
    return MetricResult(metric_id=metric_id, level="L1", value=value, unit=unit, **kw)


# --------------------------------------------------------------------------
# 1-2. binary potential energy curves
# --------------------------------------------------------------------------
@dataclass
class PECResult:
    pair: str
    r: np.ndarray
    energy: np.ndarray            # eV, referenced to the largest separation
    force_analytic: np.ndarray    # eV/A along the bond, on atom 1
    force_numeric: np.ndarray     # -dE/dr by central differences
    r_min: float                  # position of the PEC minimum
    e_min: float
    monotonic_repulsive: bool
    max_jump_ev: float                # diagnostic only: dominated by the repulsive wall
    max_force_mismatch: float         # global relative energy/force derivative agreement
    convergence_ratio: float          # residual(2h) / residual(h); ~4 for a smooth curve
    n_nonfinite: int
    #: r-interval the convergence ratio is measured over: the ends are trimmed so
    #: the h and 2h grids cover the same physical region (see :func:`binary_pec`)
    convergence_window_A: tuple[float, float] = (float("nan"), float("nan"))
    #: the ``h`` residual restricted to that same window; the ratio's denominator
    interior_force_mismatch: float = float("nan")
    #: measured energy noise of this dimer, expressed like the residuals above
    #: (relative to max |F| on the curve); the convergence-ratio test is only
    #: meaningful when the residual sits above it
    noise_residual: float = 0.0
    #: set when the calculator could not evaluate this curve at all.  A dimer is
    #: the smallest system a model ever sees, and some deployment artefacts
    #: cannot handle it (DPA-4's AOTInductor archive trips a baked-in dynamic
    #: shape guard at n_atoms = 2).  That is a packaging limit, not a PES defect,
    #: so it is reported as an analysis error rather than a model failure.
    error: str | None = None


def binary_pec(
    model: Model,
    a: str,
    b: str,
    *,
    r_min: float = 1.0,
    r_max: float = 8.0,
    dr: float = 0.02,
    box: float = 30.0,
) -> PECResult:
    """Two-atom energy curve in a non-periodic box.

    Continuity is judged two ways, because neither alone is reliable on a curve
    that spans a hard repulsive wall:

    * ``max_force_mismatch`` -- analytic force against the central-difference
      derivative of the sampled energy, relative to the largest force on the
      grid.  For a healthy model this is pure truncation error.
    * ``convergence_ratio`` -- the same residual recomputed on the ``2h``
      subgrid, divided by the ``h`` residual.  Central differences are
      ``O(h^2)``, so a genuinely smooth curve gives ~4; a discontinuity leaves a
      residual that does not shrink with ``h`` and drives the ratio towards 1.
      This is what makes the continuity test scale-free.  It is measured over
      ``convergence_window_A``, the grid with three coarse steps trimmed from
      each end, because the innermost differentiable point is not the same
      physical point on the ``h`` and ``2h`` grids -- see the comment at the
      residual definition below.

    An absolute energy-jump threshold is deliberately *not* used: at 2 A a Te-Te
    pair is still deep in its repulsive wall, where a perfectly smooth curve
    legitimately changes by ~0.5 eV between adjacent 0.02 A samples.
    """
    r = np.arange(r_min, r_max + 0.5 * dr, dr)
    energies = np.full(r.size, np.nan)
    forces = np.full(r.size, np.nan)
    first_error: str | None = None
    for k, d in enumerate(r):
        atoms = Atoms(
            symbols=[a, b],
            positions=[[0.0, 0.0, 0.0], [d, 0.0, 0.0]],
            cell=[box, box, box],
            pbc=False,
        )
        atoms.calc = model.calculator
        try:
            energies[k] = float(atoms.get_potential_energy())
            forces[k] = float(np.asarray(atoms.get_forces())[1, 0])  # along +x on atom 1
        except Exception as exc:  # a calculator that cannot do a dimer is data
            if first_error is None:
                first_error = f"{type(exc).__name__}: {exc}"

    if first_error is not None and not np.isfinite(energies).any():
        return PECResult(
            pair=pair_key(a, b), r=r, energy=energies, force_analytic=forces,
            force_numeric=np.full(r.size, np.nan),
            r_min=float("nan"), e_min=float("nan"), monotonic_repulsive=False,
            max_jump_ev=float("nan"), max_force_mismatch=float("nan"),
            convergence_ratio=float("nan"), n_nonfinite=int(r.size),
            error=first_error,
        )

    # Measure this curve's own numerical noise: a rigid translation plus rotation
    # of a two-atom cluster is an exact symmetry, so any spread is rounding.  The
    # convergence-ratio test below divides two residuals, which amplifies noise,
    # and a float32 model would otherwise be reported as discontinuous.
    rng = np.random.default_rng(0)
    probe_d = float(r[len(r) // 3])
    ref = Atoms(symbols=[a, b], positions=[[0.0, 0.0, 0.0], [probe_d, 0.0, 0.0]],
                cell=[box, box, box], pbc=False)
    ref.calc = model.calculator
    try:
        e_ref = float(ref.get_potential_energy())
    except Exception:
        e_ref = float("nan")
    noise_ev = 0.0
    for _ in range(4 if np.isfinite(e_ref) else 0):
        moved = ref.copy()
        moved.rotate(float(rng.uniform(0, 360)), rng.normal(size=3), center="COM")
        moved.positions += rng.uniform(-5.0, 5.0, size=3)
        moved.calc = model.calculator
        noise_ev = max(noise_ev, abs(float(moved.get_potential_energy()) - e_ref))

    energies -= energies[-1]
    # a central difference turns an energy noise sigma into a force noise
    # sqrt(2)*sigma/(2h); expressed on the same relative scale as the residuals
    scale0 = max(np.abs(forces[np.isfinite(forces)]).max(), 1e-8) if np.isfinite(forces).any() else 1e-8
    noise_residual = float(NOISE_MARGIN * np.sqrt(2.0) * noise_ev / (2.0 * dr) / scale0)
    return diagnose_pec(pair_key(a, b), r, energies, forces,
                        noise_residual=noise_residual, error=first_error)


def diagnose_pec(
    pair: str,
    r: np.ndarray,
    energies: np.ndarray,
    forces: np.ndarray,
    *,
    noise_residual: float = 0.0,
    error: str | None = None,
) -> PECResult:
    """Everything :class:`PECResult` reports that is *derived* from the curve.

    Split out of :func:`binary_pec` so the cache can store only what is expensive
    -- the 350 dimer evaluations per curve and the noise probe -- and have every
    diagnostic recomputed on load.  Storing a criterion's *verdict* meant that
    tightening or fixing a criterion silently had no effect on any model whose
    curves were already cached, which is exactly what happened to the
    convergence-ratio fix below: five reruns reproduced the old ratios because
    ``cached_binary_pecs`` handed back the stored numbers.
    """
    energies = np.asarray(energies, dtype=float)
    forces = np.asarray(forces, dtype=float)
    dr = float(r[1] - r[0])
    n_nonfinite = int(np.sum(~np.isfinite(energies)) + np.sum(~np.isfinite(forces)))
    numeric = np.full(r.size, np.nan)
    numeric[1:-1] = -(energies[2:] - energies[:-2]) / (2.0 * dr)

    k_min = int(np.argmin(energies))
    short = r <= THRESHOLDS["pec_min_repulsion_r"]
    monotonic = bool(np.all(np.diff(energies[short]) <= 1e-9)) if short.sum() > 1 else True

    scale = max(np.abs(forces).max(), 1e-8)

    def residual(step: int, trim: float = 0.0) -> float:
        rr, ee, ff = r[::step], energies[::step], forces[::step]
        if rr.size < 5:
            return float("nan")
        h = rr[1] - rr[0]
        fn = np.full(rr.size, np.nan)
        fn[1:-1] = -(ee[2:] - ee[:-2]) / (2.0 * h)
        ok = np.isfinite(fn) & np.isfinite(ff)
        if trim > 0.0:
            ok &= (rr >= rr[0] + trim) & (rr <= rr[-1] - trim)
        return float(np.abs(fn[ok] - ff[ok]).max() / scale) if ok.any() else float("inf")

    # The tolerance check keeps the whole grid: "does the analytic force match the
    # sampled curve" is a question about the curve the model actually exposes.
    mismatch = residual(1)
    # The *ratio* must not.  A central difference at the innermost differentiable
    # grid point straddles r_min itself -- the steepest part of the wall and the
    # edge of the sampled interval -- and that point moves outward when h doubles
    # (r_min + h vs r_min + 2h).  Taking a max over the full grid therefore
    # divides a boundary-dominated numerator by an interior-dominated denominator
    # and stops measuring the convergence order at all: it reported NequIP-OAM-L's
    # Te-Te curve at 2.4985 where the interior value is 3.84, and MACE-MPA-0's
    # Ge-Ge at 2.88 against an interior 3.92.  Excluding three coarse-grid steps
    # from each end makes both grids see the same physical interval.
    trim = 3.0 * 2.0 * dr
    fine_interior = residual(1, trim)
    coarse_interior = residual(2, trim)
    convergence_ratio = (
        float(coarse_interior / fine_interior)
        if fine_interior > 0 and np.isfinite(coarse_interior) and np.isfinite(fine_interior)
        else float("nan")
    )
    convergence_window = (float(r[0] + trim), float(r[-1] - trim))

    return PECResult(
        pair=pair,
        r=r,
        energy=energies,
        force_analytic=forces,
        force_numeric=numeric,
        r_min=float(r[k_min]),
        e_min=float(energies[k_min]),
        monotonic_repulsive=monotonic,
        max_jump_ev=float(np.abs(np.diff(energies)).max()),
        max_force_mismatch=mismatch,
        convergence_ratio=convergence_ratio,
        n_nonfinite=n_nonfinite,
        convergence_window_A=convergence_window,
        interior_force_mismatch=fine_interior,
        noise_residual=noise_residual,
        error=error,
    )


def all_binary_pecs(model: Model, species: Sequence[str], **kwargs) -> dict[str, PECResult]:
    out: dict[str, PECResult] = {}
    for a, b in combinations_with_replacement(sorted(species), 2):
        try:
            out[pair_key(a, b)] = binary_pec(model, a, b, **kwargs)
        except Exception as exc:  # keep the other five curves and the rest of L1
            r = np.arange(kwargs.get("r_min", 1.0),
                          kwargs.get("r_max", 8.0) + 0.5 * kwargs.get("dr", 0.02),
                          kwargs.get("dr", 0.02))
            nan = np.full(r.size, np.nan)
            out[pair_key(a, b)] = PECResult(
                pair=pair_key(a, b), r=r, energy=nan, force_analytic=nan,
                force_numeric=nan, r_min=float("nan"), e_min=float("nan"),
                monotonic_repulsive=False, max_jump_ev=float("nan"),
                max_force_mismatch=float("nan"), convergence_ratio=float("nan"),
                n_nonfinite=int(r.size), error=f"{type(exc).__name__}: {exc}",
            )
    return out


def pec_metrics(pecs: Mapping[str, PECResult]) -> list[MetricResult]:
    results: list[MetricResult] = []
    for key, pec in pecs.items():
        if pec.error is not None:
            results.append(_result(
                f"L1.pec.{key}",
                {"r_min_A": None, "e_min_eV": None},
                "eV",
                status=Status.ERROR,
                failure_reason=pec.error,
                extra={"note": "the calculator could not evaluate a two-atom system"},
            ))
            continue
        checks = {
            "no monotonic short-range repulsion": not pec.monotonic_repulsive,
            "non-finite energy or force on the curve": pec.n_nonfinite > 0,
            "energy/force derivative mismatch": pec.max_force_mismatch > THRESHOLDS["pec_force_mismatch_rel"],
            # Only meaningful while the residual is above the curve's own noise
            # floor: below it the ratio is noise/noise and carries no signal.
            "derivative residual does not converge as O(h^2): possible discontinuity": (
                np.isfinite(pec.convergence_ratio)
                and pec.convergence_ratio < THRESHOLDS["pec_convergence_ratio"]
                and pec.interior_force_mismatch > pec.noise_residual
            ),
        }
        reasons = [name for name, failed in checks.items() if failed]
        results.append(
            _result(
                f"L1.pec.{key}",
                {"r_min_A": pec.r_min, "e_min_eV": pec.e_min},
                "eV",
                status=Status.FAILED if reasons else Status.OK,
                failure_reason="; ".join(reasons) or None,
                extra={
                    "monotonic_repulsive": pec.monotonic_repulsive,
                    "max_force_mismatch_rel": pec.max_force_mismatch,
                    "interior_force_mismatch_rel": pec.interior_force_mismatch,
                    "derivative_convergence_ratio": pec.convergence_ratio,
                    "convergence_window_A": list(pec.convergence_window_A),
                    "noise_residual": pec.noise_residual,
                    "noise_residual_note": (
                        "measured dimer energy noise propagated through the same central "
                        "difference; the convergence-ratio check is skipped below it"
                    ),
                    "convergence_ratio_note": (
                        "residual(2h)/residual(h) over convergence_window_A; ~4 means the "
                        "residual is pure O(h^2) truncation and the curve is smooth.  The "
                        "window excludes the ends because the innermost differentiable "
                        "grid point straddles r_min and moves when h doubles"
                    ),
                    "max_energy_jump_eV": pec.max_jump_ev,
                    "max_energy_jump_note": (
                        "diagnostic only: dominated by the repulsive wall, not a continuity test"
                    ),
                    "n_nonfinite": pec.n_nonfinite,
                },
            )
        )
    return results


# --------------------------------------------------------------------------
# 3. symmetry / conservation checks
# --------------------------------------------------------------------------
def invariance_checks(
    model: Model, atoms: Atoms, *, seed: int = 0, noise_eV: float | None = None
) -> list[MetricResult]:
    """Translation, rotation, net force and net torque of a single frame.

    ``noise_eV`` is the measured energy noise floor (:func:`energy_noise`); the
    invariance tolerances are raised to it when it exceeds the float64-era
    absolute thresholds, so a float32 model is judged against its own precision
    instead of being failed for rounding.
    """
    rng = np.random.default_rng(seed)
    base = model.fresh_atoms(atoms)
    e0 = float(base.get_potential_energy())
    f0 = np.asarray(base.get_forces())

    shifted = atoms.copy()
    shifted.positions += rng.uniform(-3.0, 3.0, size=3)
    d_translation = abs(model.energy(shifted) - e0)

    # rotation is only a symmetry for a non-periodic cluster
    cluster = atoms.copy()
    cluster.set_pbc(False)
    cluster.center(vacuum=8.0)
    e_cluster = model.energy(cluster)
    rotated = cluster.copy()
    rotated.rotate(37.0, rng.normal(size=3), center="COM")
    d_rotation = abs(model.energy(rotated) - e_cluster)

    net_force = float(np.linalg.norm(f0.sum(axis=0)) / len(atoms))
    positions = atoms.get_positions() - atoms.get_center_of_mass()
    net_torque = float(np.linalg.norm(np.cross(positions, f0).sum(axis=0)) / len(atoms))

    return invariance_metrics(
        d_translation=d_translation, d_rotation=d_rotation, net_force=net_force,
        net_torque=net_torque, n_atoms=len(atoms), noise_eV=noise_eV,
    )


def invariance_metrics(
    *,
    d_translation: float,
    d_rotation: float,
    net_force: float,
    net_torque: float,
    n_atoms: int,
    noise_eV: float | None,
) -> list[MetricResult]:
    """Judge the invariance measurements of :func:`invariance_checks`.

    Split from the measurement so an evaluation pack can be re-judged on a
    machine without the model: the four numbers and the noise floor are all the
    verdict depends on.
    """
    noise = float(noise_eV) if noise_eV is not None else 0.0
    tol_translation = max(THRESHOLDS["translation_energy_tol"], NOISE_MARGIN * noise)
    tol_rotation = max(THRESHOLDS["rotation_energy_tol"], NOISE_MARGIN * noise)
    # a force noise of sigma_E / (bond length) per atom is the same rounding seen
    # through the derivative; 1 A is the right order for a covalent solid
    tol_net_force = max(THRESHOLDS["net_force_tol"], NOISE_MARGIN * noise / n_atoms)

    return [
        _result(
            "L1.translation_invariance",
            d_translation,
            "eV",
            status=Status.FAILED if d_translation > tol_translation else Status.OK,
            extra={"tolerance_eV": tol_translation, "measured_energy_noise_eV": noise},
        ),
        _result(
            "L1.rotation_invariance",
            d_rotation,
            "eV",
            status=Status.FAILED if d_rotation > tol_rotation else Status.OK,
            extra={"note": "evaluated on a non-periodic cluster; PBC breaks rotational symmetry",
                   "tolerance_eV": tol_rotation, "measured_energy_noise_eV": noise},
        ),
        _result(
            "L1.net_force",
            net_force,
            "eV/A/atom",
            status=Status.FAILED if net_force > tol_net_force else Status.OK,
            extra={"tolerance_eV_per_A_per_atom": tol_net_force},
        ),
        _result("L1.net_torque", net_torque, "eV/atom", status=Status.OK),
    ]


def force_consistency(
    model: Model,
    atoms: Atoms,
    *,
    n_atoms_probe: int = 8,
    delta: float | None = None,
    seed: int = 0,
    noise_eV: float | None = None,
) -> MetricResult:
    """Analytic forces vs central-difference ``-dE/dx`` on a random atom subset.

    With a measured energy noise floor ``noise_eV`` the step is widened towards
    the optimum for that noise (central differences trade truncation error
    ``O(delta^2)`` against noise ``O(sigma/delta)``) and the tolerance absorbs
    the residual ``sigma / (2 delta)`` force noise.  Both are recorded, so a
    loosened tolerance is visible rather than implicit.
    """
    rng = np.random.default_rng(seed)
    noise = float(noise_eV) if noise_eV is not None else 0.0
    if delta is None:
        # sigma^(1/3) is the standard optimum scaling; clipped to a range where
        # anharmonicity is still negligible for a covalent solid
        delta = float(np.clip((noise ** (1.0 / 3.0)) if noise > 0 else 1e-3, 1e-3, 2e-2))
    probe = rng.choice(len(atoms), size=min(n_atoms_probe, len(atoms)), replace=False)
    analytic = np.asarray(model.fresh_atoms(atoms).get_forces())[probe]
    numeric = np.zeros_like(analytic)
    for row, index in enumerate(probe):
        for axis in range(3):
            plus, minus = atoms.copy(), atoms.copy()
            plus.positions[index, axis] += delta
            minus.positions[index, axis] -= delta
            numeric[row, axis] = -(model.energy(plus) - model.energy(minus)) / (2.0 * delta)
    scale = max(np.abs(analytic).max(), 1e-8)
    rel = float(np.abs(analytic - numeric).max() / scale)
    return force_consistency_metric(rel=rel, delta=delta, n_probed=int(len(probe)),
                                    scale=float(scale), noise_eV=noise)


def force_consistency_metric(
    *, rel: float, delta: float, n_probed: int, scale: float, noise_eV: float | None
) -> MetricResult:
    """Judge a measured analytic-vs-finite-difference force residual."""
    noise = float(noise_eV) if noise_eV is not None else 0.0
    tol = max(THRESHOLDS["force_consistency_rel"],
              NOISE_MARGIN * noise / (2.0 * delta) / scale)
    return _result(
        "L1.force_vs_dE_dx",
        rel,
        "-",
        status=Status.FAILED if rel > tol else Status.OK,
        extra={"delta_A": delta, "n_atoms_probed": int(n_probed), "max_abs_force": float(scale),
               "tolerance_rel": tol, "measured_energy_noise_eV": noise},
    )


def _numerical_stress_voigt(model: Model, atoms: Atoms, eps: float) -> np.ndarray:
    """Central-difference ``(1/V) dE/d(strain)`` in Voigt order ``xx yy zz yz xz xy``.

    ASE ships :func:`ase.calculators.fd.calculate_numerical_stress`, but it asks
    the calculator for ``free_energy``, which CHGNet's ASE adapter does not
    implement -- and a check that silently exists for four models out of five is
    worse than no check.  This is the same convention, written out explicitly:
    deform the cell by ``(I + strain)``, carry the atoms with it, and divide the
    energy derivative by the undeformed volume.
    """
    volume = float(atoms.get_volume())
    cell = atoms.get_cell().array.copy()
    scaled = atoms.get_scaled_positions()
    out = np.zeros(6)
    voigt = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    for component, (i, j) in enumerate(voigt):
        energies = []
        for sign in (+1, -1):
            strain = np.eye(3)
            # off-diagonal components are symmetric shears: half on each side
            strain[i, j] += sign * eps * (1.0 if i == j else 0.5)
            strain[j, i] += sign * eps * (0.0 if i == j else 0.5)
            work = atoms.copy()
            work.set_cell(cell @ strain, scale_atoms=False)
            work.set_scaled_positions(scaled)
            energies.append(model.energy(work))
        out[component] = (energies[0] - energies[1]) / (2.0 * eps) / volume
    return out


def stress_consistency(
    model: Model, atoms: Atoms, *, eps: float | None = None, noise_eV: float | None = None
) -> MetricResult:
    """Analytic stress against ``(1/V) dE/d(strain)`` for all six Voigt components.

    The finite-difference strain and the pass tolerance both widen with the
    model's measured energy noise (:func:`energy_noise`), so a float32 network is
    judged against its own precision rather than against a float64-era absolute.
    """
    noise = float(noise_eV) if noise_eV is not None else 0.0
    if eps is None:
        eps = float(np.clip((noise ** (1.0 / 3.0)) * 1e-2 if noise > 0 else 1e-5, 1e-5, 2e-3))

    work = model.fresh_atoms(atoms)
    analytic = np.asarray(work.get_stress(voigt=True), dtype=float)
    numeric = _numerical_stress_voigt(model, atoms, eps)
    return stress_consistency_metric(analytic=analytic, numeric=numeric, eps=eps,
                                     volume=float(work.get_volume()), noise_eV=noise)


def stress_consistency_metric(
    *, analytic, numeric, eps: float, volume: float, noise_eV: float | None
) -> MetricResult:
    """Judge measured analytic and finite-difference Voigt stresses."""
    noise = float(noise_eV) if noise_eV is not None else 0.0
    analytic = np.asarray(analytic, dtype=float)
    numeric = np.asarray(numeric, dtype=float)
    scale = max(np.abs(analytic).max(), 1e-8)
    rel = float(np.abs(analytic - numeric).max() / scale)
    tol = max(THRESHOLDS["stress_consistency_rel"],
              NOISE_MARGIN * noise / (2.0 * eps) / volume / scale)
    return _result(
        "L1.stress_vs_dE_dstrain",
        rel,
        "-",
        status=Status.FAILED if rel > tol else Status.OK,
        extra={
            "tolerance_rel": tol,
            "measured_energy_noise_eV": noise,
            "analytic_voigt_eV_per_A3": analytic.tolist(),
            "numeric_voigt_eV_per_A3": numeric.tolist(),
            "voigt_order": "xx yy zz yz xz xy",
            "eps": eps,
        },
    )


# --------------------------------------------------------------------------
# 4. unrelaxed lattice scan
# --------------------------------------------------------------------------
@dataclass
class LatticeScan:
    scales: np.ndarray
    volumes: np.ndarray
    energy_per_atom: np.ndarray
    pressure_gpa: np.ndarray
    min_distance: np.ndarray


def lattice_scan(
    model: Model,
    atoms: Atoms,
    scales: Sequence[float] = tuple(np.round(np.arange(0.85, 1.1501, 0.01), 3)),
) -> tuple[LatticeScan, list[MetricResult]]:
    """Isotropic, *unrelaxed* linear scaling scan -- the L1 curvature check."""
    from ase.units import GPa

    from .neighbors import neighbour_list

    scales = np.asarray(list(scales), dtype=float)
    energies = np.full(scales.size, np.nan)
    volumes = np.full(scales.size, np.nan)
    pressures = np.full(scales.size, np.nan)
    dmin = np.full(scales.size, np.nan)
    for k, s in enumerate(scales):
        scaled = atoms.copy()
        scaled.set_cell(atoms.cell.array * s, scale_atoms=True)
        try:
            e, _, stress = model.energy_forces_stress(scaled)
        except Exception:
            continue
        energies[k] = e / len(scaled)
        volumes[k] = scaled.get_volume()
        pressures[k] = -float(np.trace(stress) / 3.0) / GPa
        d = neighbour_list(scaled, 4.0, "d")
        dmin[k] = float(np.min(d)) if len(d) else np.nan

    scan = LatticeScan(scales, volumes, energies, pressures, dmin)
    return scan, lattice_scan_metrics(scan)


def lattice_scan_metrics(scan: LatticeScan) -> list[MetricResult]:
    """Judge a stored lattice scan (see :func:`lattice_scan`)."""
    scales = np.asarray(scan.scales, dtype=float)
    energies = np.asarray(scan.energy_per_atom, dtype=float)
    ok = np.isfinite(energies)
    # A smooth E(scale) curve is strongly curved near full compression, so an
    # absolute second-difference threshold would flag physics.  A genuine kink
    # instead shows up as a single outlier against the rest of the curve.
    if ok.sum() > 4:
        second = np.abs(np.diff(energies[ok], n=2))
        median = float(np.median(second))
        smooth_ratio = float(second.max() / median) if median > 0 else float("inf")
        max_second_mev = float(second.max()) * 1000.0
    else:
        smooth_ratio, max_second_mev = np.nan, np.nan
    compression_rises = bool(np.all(np.diff(energies[ok][: max(1, ok.sum() // 3)]) < 0)) if ok.sum() > 3 else False
    results = [
        _result(
            "L1.lattice_scan_finite",
            int(ok.sum()),
            "count",
            status=Status.FAILED if ok.sum() < scales.size else Status.OK,
            extra={"n_requested": int(scales.size)},
        ),
        _result(
            "L1.lattice_scan_smoothness",
            smooth_ratio,
            "max/median |d2E|",
            status=(
                Status.FAILED
                if np.isfinite(smooth_ratio) and smooth_ratio > THRESHOLDS["scan_smoothness_ratio"]
                else Status.OK
            ),
            extra={
                "max_second_difference_meV_per_atom": max_second_mev,
                "note": (
                    "outlier ratio of |second difference| of E(scale); a single spike means a "
                    "discontinuous PES, while uniform curvature is just compression physics"
                ),
            },
        ),
        _result(
            "L1.lattice_scan_compression_monotonic",
            compression_rises,
            "-",
            status=Status.OK if compression_rises else Status.FAILED,
            extra={"note": "E must fall monotonically over the most compressed third of the scan"},
        ),
    ]
    return results


# --------------------------------------------------------------------------
# 5-7. stability, timestep sensitivity, NVE drift
# --------------------------------------------------------------------------
def stability_run(
    model: Model,
    atoms: Atoms,
    *,
    temperature_k: float = 2000.0,
    n_steps: int = 400,
    timestep_fs: float = 3.0,
    sample_every: int = 5,
    seed: int = 0,
    label: str = "hot",
) -> tuple[MDResult, list[MetricResult]]:
    """High-temperature NVT run: does the model survive, and how close do atoms get?"""
    settings = MDSettings(
        ensemble="nvt",
        temperature_k=temperature_k,
        timestep_fs=timestep_fs,
        n_steps=n_steps,
        sample_every=sample_every,
        seed=seed,
        protocol_id=f"mlip_nvt_{int(temperature_k)}K_{label}",
        description=f"L1 stability probe at {temperature_k:g} K",
    )
    md = run_md(atoms, model, settings)
    return md, stability_metrics(md, temperature_k=temperature_k)


def stability_metrics(md: MDResult, *, temperature_k: float) -> list[MetricResult]:
    """Judge a stored high-temperature stability run (see :func:`stability_run`)."""
    dmin = float(np.nanmin(md.min_distance)) if md.min_distance.size else np.nan
    return [
        _result(
            f"L1.stability_{int(temperature_k)}K.completed",
            md.completed,
            "-",
            status=Status.OK if md.completed else Status.FAILED,
            failure_reason=md.crash_reason,
            extra=md.summary(),
        ),
        _result(
            f"L1.stability_{int(temperature_k)}K.min_distance",
            dmin,
            "A",
            status=Status.FAILED if np.isfinite(dmin) and dmin < THRESHOLDS["min_distance_floor"] else Status.OK,
            extra={"note": "closest approach over the whole run"},
        ),
        _result(
            f"L1.stability_{int(temperature_k)}K.mean_temperature",
            float(np.nanmean(md.temperature)) if md.temperature.size else np.nan,
            "K",
            status=Status.OK,
            extra={"target_K": temperature_k},
        ),
    ]


def nve_metrics(
    md: MDResult,
    *,
    reference_drift: float | None = None,
    timestep_fs: float | None = None,
) -> list[MetricResult]:
    """Drift metrics from an existing NVE run.

    ``reference_drift`` (meV/atom/ps, measured from the AIMD NVE preflight)
    upgrades the check from a bare threshold to a tier-A comparison against a
    run with the identical cell, density, temperature and timestep.
    """
    dt = timestep_fs if timestep_fs is not None else md.settings.timestep_fs
    drift = md.drift_mev_per_atom_per_ps()
    com = float(np.nanmax(md.com_drift)) if md.com_drift.size else np.nan
    return [
        _result(
            f"L1.nve_drift.dt={dt:g}fs",
            drift,
            "meV/atom/ps",
            status=Status.FAILED if not np.isfinite(drift) or abs(drift) > THRESHOLDS["nve_drift_tol"] else Status.OK,
            evidence_tier=EvidenceTier.A if reference_drift is not None else EvidenceTier.S,
            extra={
                "aimd_reference_drift_meV_per_atom_per_ps": reference_drift,
                "ratio_to_reference": (
                    float(abs(drift) / abs(reference_drift)) if reference_drift not in (None, 0.0) else None
                ),
                **md.summary(),
            },
        ),
        _result(
            f"L1.com_drift.dt={dt:g}fs",
            com,
            "A",
            status=Status.FAILED if np.isfinite(com) and com > THRESHOLDS["com_drift_tol"] else Status.OK,
        ),
    ]


def nve_drift(
    model: Model,
    atoms: Atoms,
    *,
    temperature_k: float = 1200.0,
    n_steps: int = 333,
    timestep_fs: float = 3.0,
    sample_every: int = 3,
    seed: int = 0,
    reference_drift: float | None = None,
) -> tuple[MDResult, list[MetricResult]]:
    """Run an NVE segment and report its conserved-energy and COM drift."""
    settings = MDSettings(
        ensemble="nve",
        thermostat="none",
        temperature_k=temperature_k,
        timestep_fs=timestep_fs,
        n_steps=n_steps,
        sample_every=sample_every,
        seed=seed,
        protocol_id=f"mlip_nve_{int(temperature_k)}K_dt{timestep_fs:g}",
        description="L1 NVE drift probe",
    )
    md = run_md(atoms, model, settings)
    return md, nve_metrics(md, reference_drift=reference_drift, timestep_fs=timestep_fs)


def timestep_sensitivity(
    model: Model,
    atoms: Atoms,
    *,
    timesteps_fs: Sequence[float] = (3.0, 1.5),
    total_time_fs: float = 999.0,
    temperature_k: float = 1200.0,
    seed: int = 0,
) -> tuple[dict[float, MDResult], list[MetricResult]]:
    """Same physical duration at two timesteps; drift must not depend on ``dt``."""
    runs: dict[float, MDResult] = {}
    for dt in timesteps_fs:
        n_steps = int(round(total_time_fs / dt))
        settings = MDSettings(
            ensemble="nve",
            thermostat="none",
            temperature_k=temperature_k,
            timestep_fs=dt,
            n_steps=n_steps,
            sample_every=max(1, n_steps // 60),
            seed=seed,
            protocol_id=f"mlip_nve_dt{dt:g}",
            description="L1 timestep sensitivity",
        )
        runs[dt] = run_md(atoms, model, settings)
    return runs, timestep_metrics(runs, total_time_fs=total_time_fs)


def timestep_metrics(runs: Mapping[float, MDResult], *, total_time_fs: float) -> list[MetricResult]:
    """Judge stored NVE runs of equal duration at different timesteps."""
    drifts = {dt: r.drift_mev_per_atom_per_ps() for dt, r in runs.items()}
    finite = [d for d in drifts.values() if np.isfinite(d)]
    spread = float(max(finite) - min(finite)) if len(finite) > 1 else float("nan")
    results = [
        _result(
            "L1.timestep_sensitivity.drift_spread",
            spread,
            "meV/atom/ps",
            status=Status.FAILED if np.isfinite(spread) and spread > THRESHOLDS["nve_drift_tol"] else Status.OK,
            extra={
                "drift_by_dt": {f"{dt:g} fs": d for dt, d in drifts.items()},
                "total_time_fs": total_time_fs,
                "note": "3.0 fs is the project AIMD timestep; 1.5 fs is the control",
            },
        )
    ]
    for dt, run in runs.items():
        results.append(
            _result(
                f"L1.timestep_sensitivity.completed.dt={dt:g}fs",
                run.completed,
                "-",
                status=Status.OK if run.completed else Status.FAILED,
                failure_reason=run.crash_reason,
                extra=run.summary(),
            )
        )
    return results


# --------------------------------------------------------------------------
# single-frame probe measurements, for evaluation packs
# --------------------------------------------------------------------------
PROBE_METRICS = (
    "L1.translation_invariance",
    "L1.rotation_invariance",
    "L1.net_force",
    "L1.net_torque",
    "L1.force_vs_dE_dx",
    "L1.stress_vs_dE_dstrain",
)


def probe_record(
    results: Sequence[MetricResult] | Sequence[Mapping],
    *,
    n_atoms: int,
    volume_A3: float,
    noise_eV: float,
) -> dict[str, object]:
    """The raw measurements behind the single-frame checks, without verdicts.

    Accepts live :class:`MetricResult` objects (``run_l1.py`` writes this next to
    its MD probes) or their JSON form (backfilling a pack from an existing
    ``L1_<key>.json``); both carry the same numbers, so the record is identical.
    :func:`probe_metrics` turns it back into judged results.
    """
    rows = {}
    for r in results:
        d = r if isinstance(r, Mapping) else r.to_dict()
        rows[d["metric_id"]] = d
    missing = [m for m in PROBE_METRICS if m not in rows]
    if missing:
        raise KeyError(f"probe metrics missing: {missing}")
    force = rows["L1.force_vs_dE_dx"]
    stress = rows["L1.stress_vs_dE_dstrain"]
    return {
        "n_atoms": int(n_atoms),
        "volume_A3": float(volume_A3),
        "noise_eV": float(noise_eV),
        "d_translation_eV": float(rows["L1.translation_invariance"]["value"]),
        "d_rotation_eV": float(rows["L1.rotation_invariance"]["value"]),
        "net_force_eV_per_A_per_atom": float(rows["L1.net_force"]["value"]),
        "net_torque_eV_per_atom": float(rows["L1.net_torque"]["value"]),
        "force": {
            "rel": float(force["value"]),
            "delta_A": float(force["extra"]["delta_A"]),
            "n_atoms_probed": int(force["extra"]["n_atoms_probed"]),
            "max_abs_force": float(force["extra"]["max_abs_force"]),
        },
        "stress": {
            "analytic_voigt_eV_per_A3": [float(v) for v in stress["extra"]["analytic_voigt_eV_per_A3"]],
            "numeric_voigt_eV_per_A3": [float(v) for v in stress["extra"]["numeric_voigt_eV_per_A3"]],
            "eps": float(stress["extra"]["eps"]),
        },
    }


def probe_metrics(record: Mapping) -> list[MetricResult]:
    """Re-judge a :func:`probe_record` with the current thresholds."""
    noise = float(record["noise_eV"])
    out = invariance_metrics(
        d_translation=float(record["d_translation_eV"]),
        d_rotation=float(record["d_rotation_eV"]),
        net_force=float(record["net_force_eV_per_A_per_atom"]),
        net_torque=float(record["net_torque_eV_per_atom"]),
        n_atoms=int(record["n_atoms"]),
        noise_eV=noise,
    )
    f = record["force"]
    out.append(force_consistency_metric(rel=float(f["rel"]), delta=float(f["delta_A"]),
                                        n_probed=int(f["n_atoms_probed"]),
                                        scale=float(f["max_abs_force"]), noise_eV=noise))
    s = record["stress"]
    out.append(stress_consistency_metric(analytic=s["analytic_voigt_eV_per_A3"],
                                         numeric=s["numeric_voigt_eV_per_A3"], eps=float(s["eps"]),
                                         volume=float(record["volume_A3"]), noise_eV=noise))
    return out


# --------------------------------------------------------------------------
# gate
# --------------------------------------------------------------------------
def l1_gate(results: ResultSet | Sequence[MetricResult]) -> dict[str, object]:
    """Turn L1 into the go/no-go decision for the expensive levels."""
    items = list(results)
    failed = [r.metric_id for r in items if r.status is Status.FAILED]
    return {
        "pass": not failed,
        "n_checks": len(items),
        "n_failed": len(failed),
        "failed_metrics": failed,
        "decision": (
            "L2-L5 may run"
            if not failed
            else "L1 FAILED: do not spend compute on L4/L5 until these are resolved"
        ),
    }
