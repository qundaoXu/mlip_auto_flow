# Phase-one engineering guide: tiered L0–L5 diagnosis of MLIPs

[中文](validation_methods_L0-L5.md) · **English**

> Version v0.2, 2026-08-28
> Phase-one subjects: MACE-MPA-0 + GST225
> Background, the full list of validation methods and literature cases: `MLMD_AIMD_validation_summary.md`.

## 1. Goals of phase one

Phase one does not build a leaderboard and does not retrain potentials automatically. The goal is a reproducible diagnosis loop:

```text
read reference data and protocols
→ call MACE-MPA-0
→ run the L0–L5 tiered tasks
→ output values, evidence tiers and failing structures
→ report training-data gaps and what to add
```

At the end of phase one the project should have:

1. one format for structures, protocols and results;
2. a MACE-MPA-0 adapter pinned to a fixed version;
3. the core L0–L3 diagnosis for GST225, an L4 single-temperature smoke test and an L5 melt–quench pathway test;
4. every diagnosis traceable to its model, reference data, protocol and input structures;
5. a report that names the failing environments and exports a list of structures recommended for new calculations.

Out of scope for phase one: model fine-tuning, automatic DFT submission, a public leaderboard, nucleation rates, interface crystallisation speed, melting point, glass transition temperature and device-scale simulation.

## 2. Inputs to phase one

### 2.1 Model under test

The official `medium-mpa-0` checkpoint. MACE-MPA-0 covers 89 elements, is trained on MPTrj + sAlex, and its native level of theory is labelled PBE+U; MACE must be at least 0.3.10. Runs must additionally pin:

- the `mace-torch`, PyTorch, ASE and CUDA versions;
- the checkpoint file hash;
- `default_dtype=float64`;
- `dispersion=False`, consistent with the dispersion-free PBE reference;
- the device, random seeds and determinism settings.

Sources: [MACE repository](https://github.com/ACEsuit/mace), [MACE foundation models](https://github.com/ACEsuit/mace-foundations).

The theory MACE-MPA-0 was trained on is not exactly the protocol of this project's PBE reference. Phase one treats it as a zero-shot model under test and never takes its native energies as the reference truth.

### 2.2 GST225 reference data

Local data lives in `GST225_ref/`. Usable right now:

- 216-atom AIMD trajectories at 2000 K, 2000→300 K, structure relaxation, 300→200 K and 300→400 K;
- `AIMD/GST225_init_noStress.xyz`: 615 sampled 216-atom structures with `REF_energy`, `REF_forces` and `Virial`, covering three cell sizes and five trajectory stages;
- eleven PBE single points of one 306-atom structure under linear lattice scaling of 0.95–1.05.

The detailed audit and the additional calculations needed are in `GST225_reference_audit.md`. The 615 frames support L0 energy/force tests; the Virial needs its sign, units and source protocol checked first. Three cell sizes cannot be treated as three independent trajectories of one state, so the current data cannot yet give a reliable reference variance.

## 3. One protocol and result format

Every task has a unique `protocol_id` and records at least:

- structure ID, composition, number of atoms, cell, initial state and data role;
- temperature, pressure, ensemble, timestep, number of steps, equilibration/production segments, thermostat and seed;
- DFT code and version, functional, +U, dispersion, pseudopotentials, cutoff, k-points, smearing and convergence thresholds;
- model name, checkpoint hash, dtype, device and software versions;
- post-processing parameters: sampling interval, bond cutoffs, ring algorithm, bin width and fit window.

Only three data roles exist:

- `training`: may be used for training;
- `diagnostic`: may be used to locate problems and choose new data;
- `protected_audit`: never used for new data, fine-tuning or hyperparameter selection.

A result contains at least: `metric_id`, value/unit, uncertainty, status, evidence tier, reference/model protocols, input structure IDs, failure reason and output files.

## 4. Evidence tiers and grading

| Tier | Meaning |
|---|---|
| A | Direct comparison with AIMD under the same protocol and system size |
| B | Static DFT/DFPT under the project protocol |
| C | External protocol, a validated reference potential, or indirect large-scale evidence |
| S | Reference-free self-consistency and stability checks |

Formal grades are used only when the reference statistics are sufficient:

```text
sigma_ref = statistical spread between independent reference trajectories
d_protocol = difference on one bridge set between two reference protocols
sigma_eff = max(sigma_ref, d_protocol / 2)
z_eff = distance(model, reference) / sigma_eff
```

`z_eff ≤ 1` is A, `1–2` is B, `2–4` is C, `>4` is F. With a single reference trajectory no formal grade is computed and the result is marked `provisional`.

## 5. Tiered diagnosis tasks

### L0: reference agreement

**Purpose**: find the environments in which the model's energies, forces and stresses fail on the target structures.

Phase-one implementation:

1. MAE, RMSE, P99 and max of energy, force and stress;
2. energies compared after removing one constant offset per composition;
3. force errors binned by Ge/Sb/Te, force magnitude, temperature stage and structure type;
4. the frames and atoms with the largest errors, with their structures kept;
5. reference protocol, native model protocol and data-leakage risk recorded.

Prerequisite: confirm the DFT protocol and data role of `GST225_init_noStress.xyz`. Energies and forces can be used for L0 directly; the Virial only after its sign, units and component order are checked. If the protocol cannot be traced, recompute a bridge set under the canonical protocol instead of all 615 frames at once.

### L1: PES sanity and trajectory stability

**Purpose**: rule out obvious pathologies before any DFT reference is needed.

Phase-one implementation:

1. six pair PECs: Ge–Ge, Ge–Sb, Ge–Te, Sb–Sb, Sb–Te, Te–Te;
2. PEC continuity, short-range repulsion and numerical consistency of energy and force derivatives;
3. translation, rotation, net-force and numerical stress-derivative checks;
4. an unrelaxed energy scan under linear lattice scaling of 0.85–1.15;
5. stability at 2000 K and the distribution of the closest interatomic distance;
6. timestep sensitivity, 3 fs against 1.5 fs;
7. NVE energy drift and centre-of-mass drift.

NaN/Inf, unphysical short bonds, an interrupted trajectory, discontinuous energies or forces, or clear non-conservation mark L1 as failed, and the expensive L4/L5 are not run.

### L2: static equilibrium

**Purpose**: check the ground state, relative energies and PES curvature.

Phase-one implementation:

1. structure relaxation of cubic and hexagonal GST225;
2. relative energy and ordering of the two phases;
3. a fixed-volume, ion-relaxed EOS for each phase;
4. equilibrium volume and bulk modulus reported.

The current 306-atom eleven-point series has no phase label and `NSW=0`. It can serve as an unrelaxed volume scan or a test fixture, not as a formal relaxed EOS.

Phonons, elastic constants, defects and surfaces/interfaces wait until the phase-one framework runs end to end.

### L3a: short-range structure and chemical order

**Purpose**: compare the local structure of liquid and amorphous GST.

Phase-one implementation:

1. six partial RDFs with peak positions, heights, widths and first minima;
2. species-resolved ADFs;
3. full coordination-number distributions for Ge, Sb and Te;
4. homopolar bond fractions;
5. the distribution of the Ge tetrahedral order parameter;
6. 1-Wasserstein and Jensen–Shannon distances.

For every state point the bond cutoffs come from the first minima of the AIMD partial RDFs and are then frozen for both AIMD and MACE-MD.

### L3b: medium-range structure and topology

**Purpose**: catch medium-range errors that RDF and ADF miss but that change crystallisation.

Phase-one implementation:

1. ring-size distribution;
2. fraction and planarity of four-membered rings;
3. ALTBC;
4. S(q) and the FSDP.

The ring definition, maximum ring size and code version are fixed. Phase one uses one ring algorithm and does not maintain several implementations.

### L4: dynamics

**Purpose**: confirm that a correct structure also comes with sensible atomic motion.

Phase one implements a single fixed-temperature liquid state point:

1. per-species MSD;
2. self-diffusion coefficients;
3. the fit window and the log–log slope inside it;
4. the difference in `log10(D)` between AIMD and MACE-MD.

The existing heating and cooling trajectories cannot be used for a strict diffusion comparison. A fixed-temperature AIMD run with a production segment after equilibration has to be added. A multi-temperature Arrhenius fit comes next.

### L5: process and phase change

**Purpose**: compare the path through a given temperature programme, not only its end state.

Phase-one implementation:

1. same-size MACE-MD along the existing 2000 K → 300 K path;
2. RDF peaks, CN, four-rings, tetrahedral Ge and volume compared as functions of time and temperature;
3. the final amorphous structures compared; crystallisation speed and nucleation rate are not computed.

The existing programme has only one trajectory, so phase-one results are marked `provisional`. Formal validation needs independent melt–quench replicas.

## 6. From diagnosis to data suggestions

Phase one does not retrain automatically, but every kind of failure produces an actionable suggestion:

| Diagnosis | Structures to add first |
|---|---|
| Short-range PEC or high-temperature stability failure | Short-range scans for the six element pairs, compressed structures, anomalous high-temperature frames |
| Large L0 tail error for an element or force range | Cluster representatives for that element, coordination and force range |
| Wrong crystal-phase order or EOS | Cubic/hexagonal, strained and fixed-volume relaxed structures |
| Wrong RDF/ADF/CN | Independent liquid/amorphous frames at that temperature and density |
| Wrong rings/ALTBC | Structures with the key four-rings, long/short bond distortion and crystallisation precursors |
| Wrong diffusion | Hopping intermediates, supercooled liquid and equilibrium frames at neighbouring temperatures |
| Wrong melt–quench path | Consecutive frames around the first divergence from the reference, not only the final anomalous frames |

An exported candidate carries its failure source, selection rationale, deduplication/cluster information, the suggested DFT protocol and the metric IDs that must be re-checked.

## 7. Order of work for GST225 phase one

1. Organise the `GST225_ref` metadata and read both the raw XDATCARs and the 615-frame labelled extxyz;
2. pin the MACE-MPA-0 checkpoint and runtime environment;
3. finish L1 first, to confirm MACE-MD runs stably;
4. run MACE on the existing eleven DFT points and AIMD paths to validate the read, compute and report chain;
5. run one set of DFT convergence tests; build a bridge set and check the existing E/F/Virial labels against the canonical protocol;
6. finish L0;
7. finish L3a/L3b, with provisional results from the existing single trajectories;
8. add fixed-temperature AIMD and finish L4;
9. add independent melt–quench runs and update the reference variance for L3/L5;
10. output failing structures and training-data suggestions.

## 8. Acceptance criteria for phase one

- One protocol drives both reading the AIMD reference and running MACE-MD;
- every result traces back to its structure/protocol/model ID;
- L0 locates the specific anomalous atoms and structures;
- an L1 failure stops pointless expensive tasks;
- L3 uses exactly the same cutoffs, bins and ring parameters on both sides;
- L4 reports its fit window and does not accept a heating or cooling trajectory as equilibrium diffusion;
- single-trajectory results are clearly marked provisional;
- the report separates model error, reference statistical noise and protocol differences;
- a deduplicated list of candidate structures for new DFT can be exported;
- the protected audit set is never read or modified to guide new data.

## 9. Phase-one engineering checklist

- [ ] Define a minimal schema for structure, protocol, model and result;
- [ ] write VASP/XDATCAR readers and extxyz conversion;
- [ ] pin and verify the MACE-MPA-0 checkpoint;
- [ ] implement one calculator adapter;
- [ ] implement L0 and L1;
- [ ] implement L2 structure relaxation and EOS;
- [ ] implement L3a/L3b for GST225;
- [ ] implement single-temperature L4;
- [ ] implement the L5 pathway comparison on the existing melt–quench;
- [ ] implement independent-trajectory uncertainties and protocol-mismatch flags;
- [ ] implement failing-structure export;
- [ ] produce the first GST225 diagnosis report.
