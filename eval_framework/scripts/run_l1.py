#!/usr/bin/env python
"""Driver: L1 PES sanity and trajectory stability (the gate for L2-L5).

    python run_l1.py [--quick] [--threads T]

The MD probes deliberately reuse the phase-1 AIMD preflight starting structure
and protocol (306 atoms, 0.0309 at/A^3, 1200 K, POTIM 3.0 fs), so the NVE drift
is a like-for-like tier-A comparison against ``H0_preflight_03_nve1200``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from mlipeval import datasets, models, paths, protocols
from mlipeval.calculators import build_model
from mlipeval.pbs import describe_environment, require_compute_node
from mlipeval.io_vasp import read_poscar
from mlipeval.cache import cached_binary_pecs
from mlipeval.l1_pes import (
    energy_noise,
    force_consistency,
    invariance_checks,
    l1_gate,
    lattice_scan,
    nve_metrics,
    pec_metrics,
    probe_record,
    stability_run,
    stress_consistency,
    timestep_sensitivity,
)
from mlipeval.md_runner import save_md
from mlipeval.schema import ResultSet

SPECIES = ("Ge", "Sb", "Te")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="short MD probes, for a smoke test")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--allow-login-node", action="store_true",
                    help="bypass the compute-node check; only for trivial smoke tests")
    ap.add_argument("--skip-md", action="store_true", help="static checks only")
    args = ap.parse_args()
    require_compute_node('L1: PEC scan and MD probes', allow_login_node=args.allow_login_node)

    model = build_model(threads=args.threads, device=args.device)
    print(f"model: {model.spec.name} sha256={model.spec.checkpoint_sha256[:16]} dtype={model.spec.dtype}")

    preflight_dir = paths.PHASE1 / datasets.PREFLIGHT["nve_1200K"]
    start = read_poscar(preflight_dir / "POSCAR")
    aimd_nve = datasets.preflight_nve_drift()
    print(
        f"AIMD NVE reference: drift {aimd_nve['drift_meV_per_atom_per_ps']:+.4f} meV/atom/ps "
        f"over {aimd_nve['total_time_ps']:.3f} ps ({aimd_nve['n_atoms']} atoms)"
    )

    results = ResultSet(
        "L1",
        model=model.spec,
        probe_structure=str(preflight_dir / "POSCAR"),
        probe_protocol=protocols.canonical_aimd().to_dict(),
        aimd_nve_reference=aimd_nve,
        quick=args.quick,
    )

    # -- 1-2. binary potential energy curves -------------------------------
    print("\n[1/7] binary potential energy curves")
    pecs = cached_binary_pecs(model, SPECIES, r_min=1.0, r_max=8.0, dr=0.02)
    results.extend(pec_metrics(pecs))
    for key, pec in pecs.items():
        print(f"      {key:6s} r_min={pec.r_min:.2f} A  E_min={pec.e_min:+.3f} eV  "
              f"repulsive={pec.monotonic_repulsive}  dF/dE={pec.max_force_mismatch:.1e}  "
              f"h^2 convergence={pec.convergence_ratio:.2f}")

    # -- 3. invariance and derivative consistency --------------------------
    print("[2/7] invariance, net force, energy/force and stress derivatives")
    # Measure the model's own energy noise first: the invariance and
    # finite-difference tolerances below are float64-era absolutes, and a float32
    # network would otherwise be failed for its rounding error rather than for
    # any symmetry violation.
    noise = energy_noise(model, start)
    print(f"      measured energy noise floor: {noise:.3e} eV "
          f"({noise / len(start) * 1e3:.3e} meV/atom)")
    results.context["measured_energy_noise_eV"] = noise
    results.extend(invariance_checks(model, start, noise_eV=noise))
    results.add(force_consistency(model, start, noise_eV=noise))
    results.add(stress_consistency(model, start, noise_eV=noise))
    # the raw numbers behind these six checks, for evaluation packs (mlipeval pack):
    # with them the verdicts can be recomputed without loading the model
    probes = probe_record(results.results, n_atoms=len(start), volume_A3=float(start.get_volume()),
                          noise_eV=noise)
    paths.data_path("l1_probes.json").write_text(json.dumps(probes, indent=2) + "\n")

    # -- 4. unrelaxed lattice scan -----------------------------------------
    print("[3/7] unrelaxed isotropic lattice scan 0.85-1.15")
    scan, scan_results = lattice_scan(model, start)
    results.extend(scan_results)
    np.savez(paths.data_path("l1_lattice_scan.npz"),
             scales=scan.scales, volumes=scan.volumes,
             energy_per_atom=scan.energy_per_atom, pressure_gpa=scan.pressure_gpa,
             min_distance=scan.min_distance)

    if not args.skip_md:
        # -- 5. high-temperature stability ---------------------------------
        n_hot = 60 if args.quick else 400
        print(f"[4/7] 2000 K NVT stability probe ({n_hot} steps)")
        hot, hot_results = stability_run(model, start, temperature_k=2000.0, n_steps=n_hot)
        results.extend(hot_results)
        save_md(hot, paths.data_path("l1_stability_2000K.npz"))
        print(f"      {json.dumps(hot.summary())}")

        # -- 6-7. timestep sensitivity and NVE drift ------------------------
        total_fs = 300.0 if args.quick else 999.0  # 999 fs = 333 steps at 3 fs, matching the AIMD preflight
        print(f"[5/7] timestep sensitivity, {total_fs:g} fs at dt = 3.0 and 1.5 fs")
        runs, ts_results = timestep_sensitivity(model, start, total_time_fs=total_fs, temperature_k=1200.0)
        results.extend(ts_results)
        for dt, run in runs.items():
            save_md(run, paths.data_path(f"l1_nve_dt{dt:g}.npz"))
            print(f"      dt={dt:g} fs -> {json.dumps(run.summary())}")

        print("[6/7] NVE drift vs the AIMD preflight")
        results.extend(nve_metrics(runs[3.0], reference_drift=aimd_nve["drift_meV_per_atom_per_ps"], timestep_fs=3.0))
        results.context["md_probes"] = {"stability_temperature_K": 2000.0, "stability_steps": n_hot,
                                        "timestep_total_time_fs": total_fs, "timesteps_fs": sorted(runs)}

    # -- gate ---------------------------------------------------------------
    print("[7/7] gate decision")
    decision = l1_gate(results)
    results.context["gate"] = decision
    out = results.save(paths.result_path("L1_" + models.current_key() + ".json"))
    print(f"\nwrote {out}\n")
    print(results.to_frame().to_string(index=False, max_colwidth=40))
    print(f"\nGATE: {decision['decision']}")
    if decision["failed_metrics"]:
        print("failed: " + ", ".join(decision["failed_metrics"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
