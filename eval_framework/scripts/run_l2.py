#!/usr/bin/env python
"""Driver: L2 static equilibrium properties (relaxation, phase order, EOS).

    python run_l2.py [--threads T] [--fmax F] [--skip-cubic-eos]

DFT side (tier B, ``gst225_pbe_canonical_v1``):
  * hexagonal EOS  -- the seven completed ``H2_hex_v*`` fixed-volume tasks;
  * cubic EOS      -- all five ``H2_cub0*_v*`` ladders (5 vacancy orderings x 7
    volumes).  The MLIP is run on each ordering and paired seed by seed, so the
    reported cubic error is a mean over orderings with its across-ordering spread
    attached rather than one arbitrary seed;
  * cubic geometry -- the completed ``H1_cubic_s0*_relax`` / ``_static`` tasks;
  * phase energies -- the four-rung ladder ``L1_mp1224375_static`` (trigonal
    P-3m1), ``L1_mp3534_static`` (the other P-3m1 stacking), ``L1_mp1224407_static``
    (monoclinic Cm) and ``H1_cubic_s0*_static``, spanning 54.7 meV/atom in DFT;
  * element references -- ``L2_{ge,sb,te}_relax`` / ``_static``: relaxed element
    geometry, and the chemical potentials that turn the ladder into formation
    energies.  That is what corrects the cubic rung for its Ge43Sb43Te108
    composition, so the hex-cubic gap is quoted composition-corrected as well
    as bare.

The 306-atom ``NSW = 0`` volume sweep in ``GST225_ref`` is not read here: an
unrelaxed scan is not an EOS.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from mlipeval import models, datasets, paths
from mlipeval.calculators import build_model
from mlipeval.pbs import describe_environment, require_compute_node
from mlipeval.l2_static import (
    EOSCurve,
    eos_ensemble_metrics,
    eos_metrics,
    fit_eos,
    formation_energy_metrics,
    load_dft_eos,
    mlip_eos,
    phase_ladder_metrics,
    phase_ordering_metrics,
    relax_structure,
    relaxation_metrics,
)
from mlipeval.schema import ResultSet

EOS_SCALES = (0.92, 0.95, 0.98, 1.00, 1.02, 1.05, 1.08)


def _composition(atoms) -> dict[str, int]:
    symbols = atoms.get_chemical_symbols()
    return {s: symbols.count(s) for s in sorted(set(symbols))}


def _save_curve(curve: EOSCurve, name: str) -> None:
    np.savez(
        paths.data_path(f"l2_eos_{name}.npz"),
        volumes=curve.volumes,
        energies=curve.energies,
        n_atoms=curve.n_atoms,
        label=curve.label,
        source=curve.source,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--allow-login-node", action="store_true",
                    help="bypass the compute-node check; only for trivial smoke tests")
    ap.add_argument("--fmax", type=float, default=0.01, help="force convergence for MLIP relaxation, eV/A")
    ap.add_argument("--cubic-seed", type=int, default=1,
                    help="the seed used for the relaxation and the phase ladder")
    ap.add_argument("--cubic-eos-seeds", default="all",
                    help="'all' (every seed with a complete DFT EOS), or a comma-separated list")
    ap.add_argument("--skip-cubic-eos", action="store_true", help="the 194-atom EOS is the expensive part")
    args = ap.parse_args()
    require_compute_node('L2: relaxations and EOS scans', allow_login_node=args.allow_login_node)

    model = build_model(threads=args.threads, device=args.device)
    print(f"model: {model.spec.name} sha256={model.spec.checkpoint_sha256[:16]}")

    results = ResultSet(
        "L2",
        model=model.spec,
        dft_protocol="gst225_pbe_canonical_v1",
        fmax_eV_per_A=args.fmax,
        eos_scales=list(EOS_SCALES),
        note=(
            "every comparison is provisional (validation_methods_L0-L5.md section 4). "
            "The cubic EOS now has five DFT references, but they are five vacancy "
            "orderings rather than five repeats, so their spread is configurational and "
            "is not usable as sigma_ref; the hexagonal side still has a single reference."
        ),
    )

    # ---------------------------------------------------------------- hex
    print("\n[1/4] hexagonal GST225: MLIP relaxation vs DFT")
    hex_atoms = datasets.hexagonal_reference("mp-1224375")
    hex_dft = datasets.dft_relaxed("L1_mp1224375_relax")
    hex_dft_static = datasets.dft_relaxed("L1_mp1224375_static")
    hex_relax = relax_structure(model, hex_atoms, label="hex_mp1224375", relax_cell=True, fmax=args.fmax)
    results.extend(relaxation_metrics(hex_relax, dft_reference=hex_dft))
    print(f"      MLIP {json.dumps(hex_relax.summary())}")
    print(f"      DFT  V/atom={hex_dft['volume_per_atom_A3']:.4f} A^3  E/atom={hex_dft_static['energy_per_atom_eV']:.5f} eV")

    # -------------------------------------------------------------- cubic
    print("[2/4] cubic GST225: MLIP relaxation vs DFT")
    seed = args.cubic_seed
    cubic_atoms = datasets.cubic_seed(seed)
    cubic_dft = datasets.dft_relaxed(f"H1_cubic_s{seed:02d}_relax")
    cubic_dft_static = datasets.dft_relaxed(f"H1_cubic_s{seed:02d}_static")
    cubic_relax = relax_structure(model, cubic_atoms, label=f"cubic_s{seed:02d}", relax_cell=True, fmax=args.fmax)
    results.extend(relaxation_metrics(cubic_relax, dft_reference=cubic_dft))
    print(f"      MLIP {json.dumps(cubic_relax.summary())}")
    print(f"      DFT  V/atom={cubic_dft['volume_per_atom_A3']:.4f} A^3  E/atom={cubic_dft_static['energy_per_atom_eV']:.5f} eV")

    # -------------------------------------------------- phase ordering
    print("[3/4] phase ordering")
    # kept for continuity with phase 1: the single cubic-vs-hexagonal sign test
    results.extend(
        phase_ordering_metrics(
            {"cubic": cubic_relax.energy_per_atom, "hexagonal": hex_relax.energy_per_atom},
            dft_energies_per_atom={
                "cubic": cubic_dft_static["energy_per_atom_eV"],
                "hexagonal": hex_dft_static["energy_per_atom_eV"],
            },
            reference_phase="hexagonal",
        )
    )

    # the full ladder: the two structures phase 1 never loaded are the middle two
    # rungs, and the gap between the two P-3m1 stackings is only 15 meV/atom
    ladder_model = {"mp1224375": hex_relax.energy_per_atom,
                    "cubic": cubic_relax.energy_per_atom}
    ladder_dft = {"mp1224375": hex_dft_static["energy_per_atom_eV"],
                  "cubic": cubic_dft_static["energy_per_atom_eV"]}
    ladder_labels = {key: label for key, _, _, label in datasets.PHASE_LADDER}
    compositions = {"mp1224375": _composition(hex_relax.relaxed), "cubic": _composition(cubic_relax.relaxed)}
    for key, source, task, label in datasets.PHASE_LADDER:
        if key in ladder_model:
            continue
        atoms = datasets.mp_structure(source)
        dft_static = datasets.dft_relaxed(task)
        relaxed = relax_structure(model, atoms, label=f"phase_{key}", relax_cell=True, fmax=args.fmax)
        results.extend(relaxation_metrics(relaxed, dft_reference=datasets.dft_relaxed(task.replace("_static", "_relax"))))
        ladder_model[key] = relaxed.energy_per_atom
        ladder_dft[key] = dft_static["energy_per_atom_eV"]
        compositions[key] = _composition(relaxed.relaxed)
        print(f"      {label}: MLIP E/atom={relaxed.energy_per_atom:.5f}  "
              f"DFT E/atom={dft_static['energy_per_atom_eV']:.5f} eV")
    results.extend(phase_ladder_metrics(ladder_model, ladder_dft, labels=ladder_labels))

    # ------------------------------------------- element references
    print("[3b/4] element references and composition-corrected formation energies")
    model_mu, dft_mu = {}, {}
    for element in datasets.ELEMENTS:
        tag = element.lower()
        dft_static = datasets.dft_relaxed(f"L2_{tag}_static")
        relaxed = relax_structure(model, datasets.element_reference(element), label=f"element_{element}",
                                  relax_cell=True, fmax=args.fmax)
        results.extend(relaxation_metrics(relaxed, dft_reference=datasets.dft_relaxed(f"L2_{tag}_relax")))
        model_mu[element] = relaxed.energy_per_atom
        dft_mu[element] = dft_static["energy_per_atom_eV"]
        print(f"      {element}: MLIP E/atom={relaxed.energy_per_atom:.5f}  "
              f"DFT E/atom={dft_mu[element]:.5f} eV")
    results.extend(formation_energy_metrics(ladder_model, ladder_dft, compositions, model_mu, dft_mu,
                                            labels=ladder_labels))

    # ----------------------------------------------------------------- EOS
    print("[4/4] fixed-volume ionic-relaxed EOS")
    dft_hex = load_dft_eos(datasets.dft_eos_tasks("hex"), label="hex_dft")
    print(f"      DFT hexagonal: {len(dft_hex.volumes)} points from {dft_hex.source}")
    if getattr(dft_hex, "skipped", None):
        print(f"      skipped: {dft_hex.skipped}")
    dft_hex_fit = fit_eos(dft_hex)
    _save_curve(dft_hex, "hex_dft")

    mlip_hex = mlip_eos(
        model, hex_relax.relaxed, label="hex_mlip", scales=EOS_SCALES, relax_ions=True, fmax=args.fmax
    )
    mlip_hex_fit = fit_eos(mlip_hex)
    _save_curve(mlip_hex, "hex_mlip")
    results.extend(eos_metrics("hex", mlip_hex_fit, dft_hex_fit))
    print(f"      DFT  hex: {json.dumps(dft_hex_fit.to_dict())}")
    print(f"      MLIP hex: {json.dumps(mlip_hex_fit.to_dict())}")

    if not args.skip_cubic_eos:
        # All five H2_cub0* EOS ladders are complete, and they are five different
        # vacancy orderings of the same cell rather than five repeats.  Running
        # the MLIP on each one and pairing seed-by-seed is what separates the
        # model's systematic error from the part that depends on the ordering it
        # happened to be handed; a single seed cannot tell those apart.
        if args.cubic_eos_seeds == "all":
            eos_seeds = [s for s in range(1, 6) if datasets.dft_eos_tasks(f"cub{s:02d}")]
        else:
            eos_seeds = [int(s) for s in args.cubic_eos_seeds.split(",") if s.strip()]

        if not eos_seeds:
            print("      DFT cubic EOS not available for any seed; skipping")
        else:
            print(f"      cubic EOS seeds with a DFT reference: {eos_seeds}")

        per_seed: dict[str, tuple] = {}
        for s in eos_seeds:
            tasks = datasets.dft_eos_tasks(f"cub{s:02d}")
            if not tasks:
                print(f"      seed {s}: no completed DFT EOS tasks, skipped")
                continue
            dft_curve = load_dft_eos(tasks, label=f"cubic_s{s:02d}_dft")
            dft_fit = fit_eos(dft_curve)
            _save_curve(dft_curve, f"cubic_s{s:02d}_dft")

            # each seed needs its own relaxed starting cell: the EOS is scanned
            # around that seed's own minimum, not around seed 1's
            if s == seed:
                start = cubic_relax.relaxed
            else:
                start = relax_structure(
                    model, datasets.cubic_seed(s), label=f"cubic_s{s:02d}",
                    relax_cell=True, fmax=args.fmax,
                ).relaxed
            mlip_curve = mlip_eos(
                model, start, label=f"cubic_s{s:02d}_mlip",
                scales=EOS_SCALES, relax_ions=True, fmax=args.fmax,
            )
            mlip_fit = fit_eos(mlip_curve)
            _save_curve(mlip_curve, f"cubic_s{s:02d}_mlip")
            per_seed[f"s{s:02d}"] = (mlip_fit, dft_fit)
            print(f"      seed {s}: DFT V0={dft_fit.v0_per_atom:.4f} B0={dft_fit.b0_gpa:.3f} | "
                  f"MLIP V0={mlip_fit.v0_per_atom:.4f} B0={mlip_fit.b0_gpa:.3f}  "
                  f"(dV0={100*(mlip_fit.v0_per_atom-dft_fit.v0_per_atom)/dft_fit.v0_per_atom:+.2f}%, "
                  f"dB0={100*(mlip_fit.b0_gpa-dft_fit.b0_gpa)/dft_fit.b0_gpa:+.2f}%)")

            # keep the primary seed's curves under the historical filenames so
            # nb03 and any saved comparison keep resolving
            if s == seed:
                _save_curve(dft_curve, "cubic_dft")
                _save_curve(mlip_curve, "cubic_mlip")

        if per_seed:
            results.extend(eos_ensemble_metrics("cubic", per_seed))
        elif eos_seeds:
            print("      no seed produced a usable paired fit")

    out = results.save(paths.result_path("L2_" + models.current_key() + ".json"))
    print(f"\nwrote {out}\n")
    print(results.to_frame().to_string(index=False, max_colwidth=44))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
