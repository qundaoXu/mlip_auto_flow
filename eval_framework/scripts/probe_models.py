#!/usr/bin/env python
"""Smoke-probe one model's ASE adapter before committing a full L0-L5 chain.

    MLIPEVAL_MODEL=<key> python probe_models.py

Builds the calculator, runs three single points on the real reference structures
(a 194-atom H1 cubic static frame, the 306-atom preflight cell, and a stretched copy),
and reports what the framework needs to know up front: does it return energy,
forces *and* stress, in what dtype, how fast, and are the numbers finite.

A model that fails here would otherwise fail four hours into a queued chain.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from mlipeval import datasets, models, paths
from mlipeval.calculators import build_model
from mlipeval.io_vasp import read_poscar
from mlipeval.pbs import describe_environment, require_compute_node


def probe(model, atoms, label: str) -> dict:
    out: dict[str, object] = {"case": label, "n_atoms": len(atoms)}
    clone = model.fresh_atoms(atoms)
    t0 = time.time()
    try:
        energy = float(clone.get_potential_energy())
        forces = np.asarray(clone.get_forces(), dtype=float)
        out.update(
            energy_per_atom=energy / len(atoms),
            forces_finite=bool(np.isfinite(forces).all()),
            max_force=float(np.abs(forces).max()),
            net_force=float(np.abs(forces.sum(axis=0)).max()),
        )
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    try:
        stress = np.asarray(clone.get_stress(voigt=False), dtype=float)
        out.update(stress=True, pressure_GPa=float(-np.trace(stress) / 3 * 160.21766))
    except Exception as exc:
        out.update(stress=False, stress_error=f"{type(exc).__name__}: {exc}")
    out["wall_s"] = round(time.time() - t0, 3)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--allow-login-node", action="store_true")
    args = ap.parse_args()
    require_compute_node("model probe", allow_login_node=args.allow_login_node)

    entry = models.current()
    print(f"=== probe {entry.key} ({entry.label}) ===")
    print(f"env      : {entry.conda_env}")
    print(f"loads    : {entry.loadable}")
    print(f"results  : {paths.RESULTS}")
    print(f"node     : {json.dumps(describe_environment())}")

    t0 = time.time()
    model = build_model(device=args.device)
    print(f"built in {time.time() - t0:.1f}s: {model.spec.name} "
          f"sha256={model.spec.checkpoint_sha256[:16]} dtype={model.spec.dtype} "
          f"device={model.spec.device}")

    labelled = datasets.labelled_diagnostic()
    frame194 = next(f for f, s in zip(labelled.frames, labelled.tags["subset"]) if s == "static_cubic")
    preflight = read_poscar(paths.PHASE1 / datasets.PREFLIGHT["nve_1200K"] / "POSCAR")
    stretched = preflight.copy()
    stretched.set_cell(stretched.cell * 1.05, scale_atoms=True)

    report = [
        probe(model, frame194, "H1_cubic static (194 atoms)"),
        probe(model, preflight, "preflight_306"),
        probe(model, stretched, "preflight_306 x1.05"),
    ]
    for row in report:
        print("  " + json.dumps(row))

    ok = all("error" not in r for r in report)
    stress_ok = all(r.get("stress") for r in report if "error" not in r)
    out = paths.result_path("probe_model.json")
    out.write_text(json.dumps(
        {"model": model.spec.to_dict(), "probes": report,
         "all_energies_ok": ok, "stress_available": stress_ok}, indent=2))
    print(f"\nwrote {out}")
    print(f"VERDICT: energies={'ok' if ok else 'FAILED'}  stress={'ok' if stress_ok else 'MISSING'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
