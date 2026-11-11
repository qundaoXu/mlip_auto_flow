# Evaluation packs

An evaluation pack moves what the cluster computed **with the model** to any machine, where L0–L2 are scored on a CPU. Model calls stay on the HPC. Scoring, grading and plotting run where the pack is opened, using the same `mlipeval` functions as the drivers. A threshold, a binning or a grading rule can therefore change without new GPU time, and every model is scored by one frozen post-processing code path.

```
HPC (GPU)                                   anywhere (CPU)
mlipeval run l0|l1|l2 --model KEY           mlipeval rescore KEY.mlippack
mlipeval pack model --model KEY  ──────►    mlipeval serve   (Workbench backend)
mlipeval pack reference  (once)  ──────►    gst225_phase1_v1.mlipref
```

## Two containers

Both are zip files with `manifest.json` at the root. The manifest lists every member with its sha256, plus a `content_sha256` over that list. Opening a pack verifies every hash. It rejects unlisted members, absolute or `..` paths, more than 10 000 members, and more than `MLIPEVAL_MAX_UNPACKED_GB` (default 8) of unpacked data.

### Reference pack `*.mlipref` (schema `mlipeval-reference/1`)

This is the frozen DFT/AIMD side of L0–L2. Build it once, on the machine that holds the full phase-1 tree:

```bash
mlipeval pack reference --out data/reference/gst225_phase1_v1.mlipref   # ≈16 MB with the diagnostic set
```

| Member | Content |
|---|---|
| `L0/h5_audit.extxyz` + `.leakage.json` | 120 H5 relabels (`protected_audit`) and their leakage screen |
| `L0/diagnostic.extxyz` + `.leakage.json` | 3726 crystalline ionic steps (`diagnostic`); `--no-diagnostic` leaves them out |
| `L1/reference.json`, `L1/probe_POSCAR` | AIMD NVE drift of `H0_preflight_03_nve1200`, the probe cell, the AIMD protocol |
| `L2/reference.json` | every DFT relax/static that L2 reads (energy, volume, cell), reference compositions, the phase ladder, and the six DFT EOS curves |

`--fallback-model KEY` is for a partial phase-1 tree such as the local mirror. A DFT value whose OUTCAR is absent is then taken from what `run_l2.py` harvested on the cluster for that model: the `l2_eos_*_dft.npz` curves and the element chemical potentials. Each such value is listed under `provenance.fallbacks`, and the manifest carries `complete: false`. Every pack scored against such a reference shows a warning. Build distribution copies on hopper, without the flag.

### Model pack `*.mlippack` (schema `mlipeval-pack/1`)

A model pack holds the model outputs and nothing derived from them:

| Member | From | Used for |
|---|---|---|
| `L0/audit_predictions.npz` | prediction cache `L0_h5_120__…` | L0 audit |
| `L0/diagnostic_predictions.npz`, `L0/diagnostic_frames.json` | `L0_diagnostic_<n>__…`; indices of the frames kept after training-overlap removal | L0 diagnostic |
| `L1/pec.npz` | `pec2__…` | dimer curves (judged again by `diagnose_pec`) |
| `L1/probes.json` | `l1_probes.json` | translation/rotation invariance, net force/torque, force and stress finite differences, with the measured noise floor |
| `L1/lattice_scan.npz`, `L1/nve_dt3.npz`, `L1/nve_dt1.5.npz`, `L1/stability_2000K.npz` | `data/models/<key>/` | lattice scan, NVE drift, timestep sensitivity, 2000 K stability |
| `L2/relax.json` | `l2_relax.json` | every relaxation L2 scores: energy, volume, cell, max force, convergence, composition |
| `L2/eos_hex.npz`, `L2/eos_cubic_sNN.npz` | `l2_eos_*_mlip.npz` | MLIP EOS curves |
| `views/*.json`, `views/l3_*.npz` … | `results/models/<key>/`, `data/models/<key>/` | display only, and the originals a rescore is compared against |

```bash
mlipeval pack model --model mace_mpa0 --out packs/          # one model
mlipeval pack model --all --out packs/                      # every model with results
mlipeval pack verify packs/mace_mpa0.mlippack               # checks only; exit 1 on errors
```

Packs can be built for models that ran before the pack format existed. When `l1_probes.json` or `l2_relax.json` is absent, the same numbers come from the result files. The manifest records this under `provenance.backfilled`. Backfilled L2 cell lengths are rounded to 1e-4 Å, because that is all the result file kept.

The manifest also binds the pack to one reference: its id, content hash and label-file sha256s. It records the fingerprint of the exact frame set each prediction file was computed on. `verify` refuses predictions that do not line up with the reference frames. A pack cannot prove that its numbers came from the checkpoint it names, because only the declared sha256 is recorded. A platform should therefore build packs in its own jobs.

## Rescoring

```bash
mlipeval rescore packs/mace_mpa0.mlippack --out rescored/mace_mpa0 --compare
```

This writes `L0_audit.json`, `L0_diagnostic.json`, `L1_<key>.json` and `L2_<key>.json` in the usual `MetricResult` schema, plus `rescore_summary.json`. Each file carries `context.rescored_from_pack`: the pack and reference hashes and any verification warnings. Training-leakage stamping is applied as in the drivers. Failure-candidate export is not run, because its de-duplication needs model descriptors; candidates still come from the HPC run.

`--compare` checks every metric against the HPC original in `views/`. On hopper (2026-11-11), with the complete reference pack, all 17 models reproduce every L0 audit, L0 diagnostic, L1 and L2 value, status and grade. Iterative EOS fits are compared at 1e-4 relative. On another machine, `al_gen3` can differ in its cubic EOS ensemble. Its seed-4 MLIP EOS curve is broken (Birch–Murnaghan fit RMSE 53 eV/atom), so the optimizer's result depends on the scipy/BLAS build. That is a defect in the curve, not in the pack. `eval_framework/scripts/check_packs.py` reruns the full round trip.

## Backend

```bash
pip install -e ".[gui]"
mlipeval serve --reference data/reference/gst225_phase1_v1.mlipref     # http://127.0.0.1:8765/workbench/
```

| Endpoint | |
|---|---|
| `GET /api/health`, `GET /api/reference` | status, and the reference pack summary |
| `GET /api/packs`, `POST /api/packs` (multipart `file`) | list packs; upload a pack (it is verified on upload) |
| `GET /api/packs/{id}` | manifest summary and verification |
| `POST /api/packs/{id}/rescore?levels=L0,L1,L2&compare=true` | rescore, store the results, return the summary and the comparison |
| `GET /api/packs/{id}/results[/{stem}]` | the result files (`L0_audit`, `L0_diagnostic`, `L1`, `L2`) |
| `GET /api/packs/{id}/workbench` | the page's per-model object (`metrics`, `gate`, `l0`, `l1`, `l2`), in the same shape as the embedded `D.models[key]` |
| `DELETE /api/packs/{id}` | |

The Workbench page served at `/workbench/` uses this API. Its *Data in* panel uploads a pack, shows the verification, rescores it, and adds the model to the selector under *Uploaded packs*. Uploaded models are not ranked, and the page shows their L0–L2 views.

Interactive API docs are at `/docs`. The server has no authentication: keep the default `127.0.0.1` binding, or reach it through an SSH tunnel. Uploaded packs and results live in `--store` (default `~/.mlipeval/workbench`).

## L2 EOS window (fixed 2026-11-11)

The DFT tasks `H2_*_v092…v108` scale the **volume** (V/V0 = 0.92–1.08). Until 2026-11-11, `mlip_eos` applied the same factors to the **cell vectors**, so every MLIP EOS spanned V/V0 = 0.78–1.26. The two Birch–Murnaghan fits therefore covered different windows, and the V0/B0 errors carried fit-range bias.

`mlip_eos` now scans in volume (`scale_kind="volume"`), and `run_l2.py` records `eos_scale_kind` in its context. L2 was rerun for all 17 models on hopper (PBS jobs 712359/712361/712364/712366). The relaxations reproduced to ≤ 0.003 meV/atom, and every EOS now fits with an RMSE below 5 meV/atom over the DFT window. The fix changed the cubic B0 errors the most. For the nine showcase models they now range from −27% to +11%, instead of uniformly −10% to −39%. The previous files are kept in `results/models/<key>/l2_linear_eos_retired_20261111/` and `data/models/<key>/l2_linear_eos_retired_20261111/`. A pack built from a pre-fix file still gets the `L2.eos.scale_kind` warning.
