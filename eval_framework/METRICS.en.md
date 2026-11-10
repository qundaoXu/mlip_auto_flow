# Metrics of the eval framework (L0–L5)

[中文](METRICS.md) · **English**

This file describes every metric **currently implemented** in `eval/scripts/mlipeval/`: which code produces it, its mathematical definition, the physical question it asks, and when it is marked `failed` or `skipped`.

The normative sources are `validation_methods_L0-L5.md` (task definitions and acceptance criteria) and `database_link.md` (protocol isolation and data roles). This file only describes **what the code actually computes**; it does not repeat parts of the specification that are not implemented yet.

Result files: `eval/results/*.json` (phase-one MACE-MPA-0) and `eval/results/models/<model_key>/*.json` (the five-model extension).

| Level | Output file | Metrics (this run) | Main question |
|---|---|---|---|
| L0 | `L0_<protocol_id>.json` | 34 | Does the potential get energies, forces and stresses right **on given structures** |
| L1 | `L1_<model>.json` | 23 | Is the PES itself **self-consistent, continuous and integrable** (reference-free, tier S) |
| L2 | `L2_<model>.json` | 13 | **Static equilibrium** of ordered phases: structure, phase order, equation of state |
| L3a | `L3a_<state>.json` | 52 × 2 states | **Short-range structure and chemical order** of disordered phases |
| L3b | `L3b_<state>.json` | 18 × 2 states | **Medium-range order and topology** of disordered phases |
| L3-u | `L3_uncertainty_<state>.json` | 33 × 2 states | A **measurable error scale** for every L3 scalar, and whether a difference is resolvable |
| L4 | `L4_<state>.json` | 12 | **Transport**: MSD and self-diffusion coefficients |
| L5 | `L5_<programme>.json` | 16 | **Process/pathway**: reproducing the melt–quench along the way, and the first divergence |
| Cross-model | `ranking.json` | 30 scored indicators | **Relative** ranking of five models against one reference |

---

## 0. Bookkeeping shared by all levels

### 0.1 One metric record (`schema.MetricResult`)

Every metric carries at least: `metric_id`, `level`, `value`, `unit`, `status`, `evidence_tier`, `grade`/`z_eff`, `reference_protocol`, `model_protocol`, `structure_ids`, `failure_reason`, `outputs`, `extra`. This is the traceability requirement (section 3 of the specification) in code: any number can answer "compared with what, under which protocol, and wrong on which structure".

### 0.2 `status`: what state a number is in

| status | Meaning |
|---|---|
| `ok` | Computed, and it is a self-consistent/reference-free verdict (or a purely descriptive quantity) |
| `provisional` | Computed, but **the reference spread `sigma_ref` is unknown**, so no formal grade is given |
| `failed` | The model, or this run, did not pass the check |
| `skipped` | A precondition is not met (for example a closed L1 gate, or an FSDP the box cannot resolve) |
| `error` | The analysis itself did not run (for example a backend that cannot evaluate a two-atom system) |

**Why most model–reference comparisons are `provisional`**: each state point or phase has only one reference trajectory so far, so `sigma_ref` cannot be measured, and L0 also lacks `d_protocol` (it needs the 24 SCFs of the bridge set). `schema.grade_metric` **refuses to invent a grade** when `n_independent_references < 2`.

### 0.3 `evidence_tier`: how hard a comparison is

| tier | Meaning | Typical metrics |
|---|---|---|
| **A** | Direct comparison with AIMD under the same protocol and size | All of L3/L4/L5; L1 NVE drift (when an AIMD preflight is available) |
| **B** | Static DFT/DFPT under the project protocol | L0 E/F; L2 volume/EOS/phase order |
| **C** | External protocol / indirect evidence | L0 stress (the reference Virial convention has only been audited, not re-derived) |
| **S** | Reference-free self-consistency and stability checks | All of L1; L0 health checks; L4 equilibrium and unwrap guards |

### 0.4 Grading formula (not yet active)

```
sigma_eff = max(sigma_ref, d_protocol / 2)
z_eff     = |distance(model, ref)| / sigma_eff
z ≤ 1 → A;  1–2 → B;  2–4 → C;  > 4 → F
```

`sigma_ref` = spread between independent reproductions on the reference side; `d_protocol` = systematic difference between two DFT protocols on the same structures. If either is missing, the result is `provisional`. Both are missing at present, so **the framework holds no formal grade**.

### 0.5 Protocol isolation and energy offsets

- Every label set has its own `protocol_id`; data from different protocols **may not be merged into one RMSE**.
- Before comparing energies, the **only** allowed correction is one constant per-atom offset within "same composition + same protocol" (`l0_reference.constant_energy_offset`). Forces and stresses get **no empirical correction**.

---

## 1. L0 — reference agreement (`l0_reference.py`)

**The question**: given 615 structures already labelled by DFT, how far are the model's energies, forces and stresses on **the same structures**? This is the only level where the model does not run its own dynamics, so its error is pure fitting error with no contribution from diverging trajectories.

### 1.1 Model health

| metric_id | Unit | Definition and meaning |
|---|---|---|
| `L0.model_eval_failures` | count | Frames on which the calculator raises. Non-zero means `failed`: the model cannot even run on the reference structures, and nothing after it is worth reading |
| `L0.nonfinite_energies` | count | Frames with NaN/Inf energy. Same as above |

### 1.2 Energy (tier B)

First fit a per-atom constant offset for each composition, `offset_c = Σ(E_model − E_ref) / Σ N`, then compare `ΔE/N = (E_model − N·offset − E_ref)/N`.

| metric_id | Unit | Meaning |
|---|---|---|
| `L0.energy_mae` / `_rmse` / `_p99` / `_max` | meV/atom | Mean / root-mean-square / 99th percentile / maximum of the per-atom energy error. An RMSE far above the MAE means a few frames dominate the error; P99 and max are the worst case, and decide whether spurious local driving forces appear in MD |
| `L0.energy_offset_spread` | meV/atom | Range of the offsets fitted for different compositions. **A protocol diagnostic**: if compositions need very different offsets, model and reference differ by more than a constant (element reference states, pseudopotentials), and the premise of the energy comparison breaks. Zero in this run (one composition) |

> Physics reminder: the absolute energy MAE cannot be read across protocols. This run's 14.7 meV/atom is the sum of the difference between MACE-MPA-0's training protocol (PBE+U, MPtrj/sAlex) and the legacy GST225 static protocol, plus model error. The two are not separated; separating them is what the bridge set is for.

### 1.3 Force (tier B)

| metric_id | Unit | Meaning |
|---|---|---|
| `L0.force_mae` / `_rmse` / `_p99` / `_max` | eV/Å | Statistics of the force **component** error. Forces need no energy zero, so this is the cleanest statement of accuracy and the safest to read across protocols |
| `L0.force_component_r2` | – | Pearson R² between reference and model force components. It measures overall correlation of direction and magnitude; high R² with a large MAE means systematic scaling, low R² means a structural error |
| `L0.force_mae.species=<X>` | eV/Å | Binned by element: which element's chemical environment was not learned (in GST, the tetrahedral/octahedral two-state behaviour of Ge is the usual suspect) |
| `L0.force_vector_mae.\|F\|=[a,b)` | eV/Å | Force-vector error binned by **reference force magnitude**. Low-\|F\| bins reflect the near-equilibrium region (phonons and relaxation), high-\|F\| bins the strongly repulsive region (stability of high-temperature MD). An error rising monotonically with \|F\| is normal; a larger error in the low-\|F\| bins means the PES near equilibrium has been smoothed out |
| `L0.force_rmse.<tag>=<group>` | eV/Å | Per-frame force RMSE binned by any frame-level tag. This run uses `stage` (1-2000K / 2-300K / 3-opt / 4-MD / 5-MD), `state_kind` (liquid / quench / amorphous_ramp / relaxed) and `density_tag` (d27/d28/d29). **A transferability metric**: a phase or density that is clearly worse is a target for new data |

### 1.4 Stress (tier C)

| metric_id | Unit | Meaning |
|---|---|---|
| `L0.stress_rmse` | meV/Å³ | RMSE of the stress tensor components. The sign, units and component order of the reference `Virial` are **not traceable to a protocol**; `io_labeled.audit_virial_convention` audits them by correlation with the model stress, and this run settles on `stress = −Virial/V` (r = 0.986; the opposite sign gives −0.986). The metric is `skipped` when the audit is inconclusive. **Diagnostic only, never scored** |

### 1.5 Worst-structure export

| metric_id | Meaning |
|---|---|
| `L0.worst_cases` | The 10 frames and 25 atoms with the largest force errors (element, frame index, error), with their structures kept. `failure_export` turns them into candidates for new calculations; frames with the `protected_audit` role are refused |

---

## 2. L1 — PES self-consistency and trajectory stability (`l1_pes.py`, all tier S)

**The question**: ignoring the reference and looking only at the model, is this a **legitimate** potential energy surface: continuous, differentiable, with F = −∇E, a repulsive core, and trajectories that integrate without blowing up? L1 is the **gate**: if `l1_gate()` finds any `failed`, no compute is spent on L4/L5 (rule 6 of the specification).

### 2.0 Noise floor (a correction relative to the specification)

`energy_noise()` measures the model's energy noise on a structure with three **strictly invariant** transformations: rigid translation, permutation of atom indices, and both combined. A float32 model on a 306-atom cell has noise of about 1e-4 eV, so an absolute threshold of 1 µeV from the float64 era would call rounding error a broken symmetry. Every invariance and finite-difference check therefore uses a tolerance of `max(absolute threshold, 5 × propagated measured noise)` and writes the tolerance it used into `extra`. **The permutation probe is essential**: models that build their graph from fractional coordinates (CHGNet) return exactly zero for a pure translation and would report zero noise.

### 2.1 Pair potential energy curves (PEC)

`L1.pec.<A>-<B>` (six curves: Ge-Ge/Ge-Sb/Ge-Te/Sb-Sb/Sb-Te/Te-Te), value = `{r_min, e_min}`, the equilibrium bond length and well depth. Two atoms in a vacuum box, r from 1.0 to 8.0 Å in 0.02 Å steps.

Checks (any one triggering means `failed`, recorded in `failure_reason`):

| Check | Threshold | Physical meaning |
|---|---|---|
| Monotonic short-range repulsion | E(r) must increase monotonically for r ≤ 1.6 Å | A potential without a repulsive core lets atoms pass through each other in high-temperature MD, a catastrophic failure |
| No NaN/Inf on the curve | — | Numerical breakdown in the extrapolation region |
| Consistent energy/force derivatives | `max\|F_analytic − (−dE/dr)\|/max\|F\| ≤ 0.05` | The force must really be the gradient of the energy |
| O(h²) convergence of the derivative residual | `residual(2h)/residual(h) ≥ 2.5` | **A dimensionless continuity test.** Central differences are O(h²), so a smooth curve gives ~4; at a discontinuity the residual does not shrink with h and the ratio tends to 1. Applied only when the residual is above the curve's own noise floor, otherwise reported as uninformative |

> Why not an absolute threshold on energy jumps: at 2 Å Te-Te is still on the repulsive wall, and neighbouring points 0.02 Å apart legitimately differ by ~0.5 eV. `max_energy_jump_eV` is recorded as a diagnostic only.

### 2.2 Symmetry and conservation

| metric_id | Unit | Definition | Physical meaning |
|---|---|---|---|
| `L1.translation_invariance` | eV | Energy change under rigid translation | Translational invariance; breaking it gives the system an absolute position and breaks momentum conservation |
| `L1.rotation_invariance` | eV | Energy change of a non-periodic cluster rotated by 37° | Rotational invariance (a periodic box breaks rotational symmetry itself, so it is tested on a cluster); breaking it breaks angular-momentum conservation |
| `L1.net_force` | eV/Å/atom | `\|Σ_i F_i\|/N` | Newton's third law, the derivative form of translational invariance. A non-zero net force accelerates the centre of mass spontaneously |
| `L1.net_torque` | eV/atom | `\|Σ_i r_i × F_i\|/N` (about the centre of mass) | A non-zero net torque is normal in a periodic system (recorded, not judged) |
| `L1.force_vs_dE_dx` | – | 8 random atoms × 3 directions, `max\|F_analytic − F_numeric\|/max\|F\|` | Whether the force equals −∂E/∂x. The difference step is set from the measured noise at its σ^(1/3) optimum |
| `L1.stress_vs_dE_dstrain` | – | Six Voigt components, central difference of `(1/V)dE/dε` against the analytic stress | Whether the stress equals the strain derivative of the energy. **Uses the framework's own differences** (`_numerical_stress_voigt`), because ASE's implementation needs `free_energy`, which CHGNet lacks; a check that works for four models out of five is worse than none |

### 2.3 Lattice scaling scan (unrelaxed)

Isotropic linear scaling from 0.85 to 1.15, 31 points.

| metric_id | Unit | Meaning |
|---|---|---|
| `L1.lattice_scan_finite` | count | Points out of 31 that were computed; one missing means `failed` (breakdown in the extrapolation region) |
| `L1.lattice_scan_smoothness` | max/median \|d²E\| | **Outlier ratio** of the second difference of E(scale); > 25 fails. A ratio rather than an absolute value, because the compressed region is strongly curved anyway and a real kink shows up as a single outlying point |
| `L1.lattice_scan_compression_monotonic` | bool | In the most compressed third, E must rise monotonically towards compression. Compression without rising energy means a false minimum on the high-density side, and MD collapses |

### 2.4 High-temperature stability

| metric_id | Unit | Meaning |
|---|---|---|
| `L1.stability_2000K.completed` | bool | Whether 400 steps of 2000 K NVT finish (no NaN, no blow-up) |
| `L1.stability_2000K.min_distance` | Å | **Closest interatomic distance** over the run. < 1.6 Å fails: a stable GST melt should never bring atoms this close, and if it does the repulsive core is too weak |
| `L1.stability_2000K.mean_temperature` | K | Whether the thermostat really holds the target temperature (diagnostic) |

### 2.5 Timestep sensitivity and NVE drift

| metric_id | Unit | Meaning |
|---|---|---|
| `L1.timestep_sensitivity.drift_spread` | meV/atom/ps | Difference in NVE drift between dt = 3.0 fs and 1.5 fs over the same physical time. > 5 fails. **This tests whether the project's 3.0 fs AIMD timestep is legitimate**: if the drift depends strongly on dt, the PES has high-frequency structure and 3 fs integration does not hold |
| `L1.timestep_sensitivity.completed.dt=<x>fs` | bool | Whether the run at each timestep completed |
| `L1.nve_drift.dt=3fs` | meV/atom/ps | Linear drift rate of the total energy over the NVE segment. \|drift\| > 5 fails. **Energy conservation is direct evidence of integrability**; with an AIMD preflight for comparison (same cell, density, temperature and dt) it is promoted to tier A |
| `L1.com_drift.dt=3fs` | Å | Maximum centre-of-mass displacement; > 1e-3 fails. The dynamical consequence of net-force conservation |

### 2.6 The gate

`l1_gate()` returns `{pass, n_checks, n_failed, failed_metrics, decision}`, written into the `context` of the L1 result file. `run_all.py` stops before L2 when it does not pass.

---

## 3. L2 — static equilibrium (`l2_static.py`, tier B/S)

**The question**: does the model get the equilibrium structure, relative stability and volume–energy relation of the **ordered phases** (cubic and hexagonal GST225) right? This is the bridge between L0 (single-point accuracy) and L3+ (disordered dynamics): a model that is accurate on single points but relaxes away from the right structure has a systematic error along the gradient.

### 3.1 Relaxation (`FrechetCellFilter` + FIRE, fmax = 0.01 eV/Å)

| metric_id | Unit | Meaning |
|---|---|---|
| `L2.relax.<label>.energy_per_atom` | eV/atom | Per-atom energy after relaxation (tier S, a model-internal quantity). Not converging to fmax means `failed` |
| `L2.relax.<label>.volume_error` | % | Deviation from the DFT-relaxed volume of the same starting structure. **The most direct reading of over- or under-binding**: negative = volume too small = bonds too strong |
| `L2.relax.<label>.cell_length_error` | % | Maximum error over the three cell lengths (per-axis values in `extra`). Read alongside the volume error, it separates overall scaling from anisotropic distortion; the c axis of the hexagonal phase (with its van der Waals-like gap) is a common failure point |

### 3.2 Phase order

| metric_id | Unit | Meaning |
|---|---|---|
| `L2.phase_order.<phase>_minus_<base>` | meV/atom | Per-atom energy difference between two phases (same composition and protocol, so constant offsets cancel). `extra` carries the DFT value and the error; **a sign opposite to DFT means `failed`**: a potential with the phase order reversed cannot crystallise correctly. This run: `cubic_minus_hexagonal = +44.9 meV/atom` (hexagonal more stable, as expected) |

### 3.3 Equation of state (EOS)

A volume scan at fixed volume with **ionic relaxation** (linear scaling 0.92–1.08, matching the phase-one `H2_*_v092..v108` tasks), fitted with a third-order Birch–Murnaghan EOS.

`load_dft_eos` **refuses tasks with `NSW ≤ 0`**: an unrelaxed volume scan is not an EOS (so the 306-atom eleven-point `NSW=0` scan in GST225_ref is not accepted).

| metric_id | Unit | Meaning |
|---|---|---|
| `L2.eos.<label>.V0` | Å³/atom | Fitted equilibrium volume; an estimate independent of the relaxed volume |
| `L2.eos.<label>.B0` | GPa | Bulk modulus = **curvature** of E(V) at V0. Much more sensitive than V0: a matching volume with B0 off by 20% means the width of the well (phonon stiffness, thermal expansion) is wrong |
| `L2.eos.<label>.V0_error` / `B0_error` | % | Relative error against the DFT fit. `extra` carries the full fit parameters of both sides (including `Bp` and the fit RMSE, the latter a self-check on whether the curve is worth fitting) |

---

## 4. L3a — short-range structure and chemical order of disordered phases (`l3a_short_range.py`, tier A)

**The question**: in amorphous and liquid GST225, are **bond lengths, coordination and chemical order** right? The core phase-one conclusions come from this level.

### 4.0 Frozen parameters (the precondition of this level)

`freeze_params_from_reference()` derives a bond cutoff for each element pair from the first minimum of the **reference (AIMD)** partial RDF, packs them into `StructureParams`, and applies them **unchanged** to the MLIP trajectory. `cutoff_diagnostics` records per pair the peak position, minimum position, minimum depth `(g_peak−g_min)/g_peak`, and whether the cutoff was measured or inherited from the total RDF.

> **An explicit choice is required when using it**: in 300 K amorphous GST, 5 of 6 pairs have a well-defined minimum (depth 0.80–0.88) and this works well. In the 2000 K liquid, 4 pairs degenerate; falling back to the total-RDF value of 4.49 Å reaches past the first shell and lifts `<CN>` to 9.8 (counting into the second shell). Inheriting with `--cutoffs-from amorphous_300K` brings the liquid `<CN>` back to 4.34. The choice is written to `cutoff_source_state` in the result JSON and **must be cited with the conclusion**.

### 4.1 Two rulers for distributions

Both are computed for every distribution, because they are sensitive to different errors:

- **1-Wasserstein `W1`** ("earth mover's distance"): sensitive to **shifts** of a distribution, in the units of its axis (Å / deg). A peak moved by 0.1 Å gives W1 ≈ 0.1.
- **Jensen–Shannon distance** (base 2, in [0,1]): sensitive to **shape and redistribution of weight**, dimensionless. A peak that stays put but gets taller and narrower barely changes W1 while JS rises.

### 4.2 Partial radial distribution functions

Six pairs `Ge-Ge, Ge-Sb, Ge-Te, Sb-Sb, Sb-Te, Te-Te`:

| metric_id | Unit | Meaning |
|---|---|---|
| `L3a.rdf.<pair>.r_peak` | Å | First-peak position = **typical bond length** of the pair |
| `L3a.rdf.<pair>.g_peak` | – | First-peak height = **relative abundance / order** of the bond. In GST the `Ge-Ge` and `Sb-Sb` peak heights read out the homopolar (wrong) bond content directly |
| `L3a.rdf.<pair>.fwhm` | Å | Full width at half maximum of the first peak = **width** of the bond-length distribution, reflecting local disorder and thermal fluctuation |
| `L3a.rdf.<pair>.wasserstein1` | Å | Shift-type difference over the whole g(r) |
| `L3a.rdf.<pair>.jensen_shannon` | – | Shape-type difference over the whole g(r) |
| `L3a.rdf_reference.<pair>.<field>` | | Reference values only, when there is no model trajectory (including `first_minimum`, the cutoff used) |

Every `extra` carries the triple `reference` / `model` / `difference`.

### 4.3 Bond-angle distributions (ADF)

Resolved by the element of the central atom (Ge / Sb / Te / all), normalised to unit area.

| metric_id | Unit | Meaning |
|---|---|---|
| `L3a.adf.<species>.jensen_shannon` | – | Shape difference of the angle distribution |
| `L3a.adf.<species>.wasserstein1` | deg | Shift difference of the angle distribution |

Physically: **the peak near 90° = defective octahedral coordination** (the parent structure of GST), **the peak near 109.5° = tetrahedral coordination** (the minority state of Ge). Their relative weight is the central question of amorphous GST structure; read the ADF together with the tetrahedral order parameter below.

### 4.4 Coordination number (CN)

Counted per frame under the frozen cutoffs; the output is the **full distribution**, not only its mean.

| metric_id | Unit | Meaning |
|---|---|---|
| `L3a.cn.<species>.mean` | – | Mean coordination. For Ge/Sb/Te, the deviation from 4/3/2 under the 8−N rule and towards octahedral bonding; `all` is the overall mean |
| `L3a.cn.<species>.jensen_shannon` | – | Distance between the **full CN distributions**. Equal means with different distributions = under- and over-coordination cancelling, a failure mode the mean hides |

### 4.5 Homopolar bonds (chemical order)

| metric_id | Unit | Meaning |
|---|---|---|
| `L3a.homopolar.<X>` | fraction of bonds | X–X bonds as a fraction of all bonds (`Ge`/`Sb`/`Te`/`total`). **The quantitative measure of chemical order**: in ideal GST225, Ge and Sb bond only to Te, so homopolar bonds are wrong bonds. They directly affect crystallisation speed, band gap and resistance contrast, and are among the most important structural quantities of a phase-change memory material |

> The core result of this run is here: MACE-MPA-0 raises Ge–Ge homopolar bonds from 0.0191 to 0.0319 (**+67%**) and total homopolar bonds by 15%, while all six RDF first-peak positions sit **within** statistical noise. **Bond lengths are learned; chemical order is not.** An independent liquid run at 1985 K reproduces the same signature, so this is a transferable systematic bias.

### 4.6 Tetrahedral order parameter (Chau–Hardwick q)

`q = 1 − (3/8) Σ_{i<j} (cosθ_ij + 1/3)²` over the 4 nearest bonded neighbours. `q = 1` is an ideal tetrahedron, `q = 0` the ideal-gas value.

| metric_id | Unit | Meaning |
|---|---|---|
| `L3a.tetrahedral_q.Ge.mean` | – | Mean tetrahedrality of Ge; `extra` also carries `fraction_q>0.8` (the fraction of Ge that is truly tetrahedral) |
| `L3a.tetrahedral_q.Ge.wasserstein1` | – | Overall difference of the q distribution |

Physics: the **octahedral ↔ tetrahedral two-state behaviour** of Ge in a-GST is at the heart of the umbrella-flip crystallisation mechanism. Get the tetrahedral Ge fraction wrong and the crystallisation kinetics will be wrong.

---

## 5. L3b — medium-range order and topology (`l3b_medium_range.py`, tier A)

**The question**: once the short range is right, is the connectivity **a few bonds away** right? Quantities at this level are the most sensitive to structures that look similar but crystallise completely differently.

### 5.1 Ring statistics

**Only one** algorithm is implemented and frozen: the **King shortest-path criterion**. For each atom i and each pair of its bonded neighbours (j, k), find the shortest j→k path that avoids i and close it with i into a ring; deduplicate by node set and truncate at `max_ring_size = 12`. (Changing the algorithm changes every ring count, so the algorithm name and maximum ring size are written into `StructureParams` and echoed in every metric.)

| metric_id | Unit | Meaning |
|---|---|---|
| `L3b.rings.size<N>` | rings/atom | Number of N-membered rings per atom (N = 3…12). **Four-rings are the proxy for cubic order in GST**: ABAB four-rings are the nucleation units on the way from amorphous to rocksalt. A shift towards larger rings = a looser, more disordered network and slower crystallisation |
| `L3b.four_ring_planarity.<reference\|model>` | Å rms out-of-plane | RMS out-of-plane deviation of four-rings (0 = perfectly planar). **Planarity is what makes a four-ring cubic-like**: the right number of badly puckered four-rings are not crystallisation precursors |

> This run: too few 3/5/6/12-rings and too many 7/10-rings. The topology shifts towards larger rings, the other face of the same chemical-order problem as the excess homopolar bonds.

### 5.2 ALTBC (angular-limited three-body correlation)

For each central atom, take every pair of bonds whose angle falls within 180° ± 25° and put their two lengths `(d1, d2)` into a 2D histogram.

| metric_id | Unit | Meaning |
|---|---|---|
| `L3b.altbc.asymmetry` | Å | Mean `\|d1 − d2\|` over nearly collinear bond pairs. **A quantitative measure of the Peierls distortion**: in the resonantly bonded GST system, collinear three-atom chains spontaneously form a long/short bond pair (off-diagonal weight). A model that over-symmetrises the octahedra piles weight onto the diagonal and gives a small asymmetry, which maps directly onto errors in electronic structure and optical contrast |

### 5.3 Structure factor S(q)

**Summed directly over the reciprocal vectors of the simulation box**, not a Fourier transform of g(r): for a 22 Å box the transform is unphysical at q < 2π/r_max (measured on this system: S(0.4) = 6.7, S(0.7) = −0.6). Neutron weights by default (coherent scattering lengths Ge 8.185 / Sb 5.57 / Te 5.80 fm).

| metric_id | Unit | Meaning |
|---|---|---|
| `L3b.fsdp.q_fsdp` / `S_fsdp` / `fwhm` | 1/Å, –, 1/Å | Position/height/width of the first sharp diffraction peak (FSDP, search window 0.8–1.7 1/Å). The FSDP is the **experimental observable of medium-range order** (quasi-periodicity at ~2π/q ≈ 6 Å). When the peak sits on the window edge, or the lower window edge is below the smallest wavevector the box can express (`2π/L`), it is marked `resolved=False` and the metric `skipped`. A 306-atom GST cell is right at this limit, so all three FSDP metrics are `skipped` in this run: **an honest report of the resolution limit**, not a failure |
| `L3b.principal_peak.q_pp` / `S_pp` | 1/Å, – | Position and height of the principal peak (~2.2 1/Å, search window 1.7–3.2). The box **resolves it well**, so it is the usable comparison on the S(q) side |

---

## 6. L3 uncertainty layer (`uncertainty.py`)

**The question**: each L3 state point produces hundreds of numbers, and without error bars "the model is wrong" cannot be told apart from "the trajectory is too short". This module provides two **measurable** error scales and states explicitly that **neither is `sigma_ref`**, so neither enters the grading.

| Scale | Definition | Question it answers |
|---|---|---|
| **Block SEM** | Cut one trajectory into 5 **consecutive** blocks (not random samples: MD frames are strongly autocorrelated, and random sampling badly underestimates the error), compute the observable on each block, and take the standard error of the block means | "Has this number converged?" (**statistical error within one chain**) |
| **Seed scatter** | Range of the same observable over two MLIP trajectories that differ only in the random seed | "Could the same model produce this difference by chance with another seed?" (**chain-to-chain fluctuation at fixed physics**, measured on the model side) |

`l3_scalars()` defines the 33 scalars tested: `cn_mean.{Ge,Sb,Te,all}`, `homopolar.{Ge,Sb,Te,total}`, `tetrahedral_q.Ge.{mean, fraction_gt_0.8}`, `rdf.<pair>.{r_peak, g_peak}` (6 pairs × 2), `rings.size3..12`, `altbc.asymmetry`.

Verdicts (`compare_with_uncertainty`, z threshold 2.0):

| verdict | Meaning |
|---|---|
| `resolved` | The difference exceeds both the combined block SEM (× z) **and** the model's own seed scatter → a real difference at the current statistics |
| `within_statistical_noise` | The difference does not exceed the combined block SEM → the trajectories are too short to tell the two sides apart at all |
| `within_seed_scatter` | Statistically significant, but no larger than the fluctuation the same model produces with another seed (the scatter is floored by the SEM, so a near-zero range between two seeds cannot fake significance) → **not attributable to model error** |
| `unavailable` | One side's estimate is missing or non-finite |

> For 300 K amorphous GST in this run, of 33 observables **18 are resolvable, 3 are significant but within the seed scatter, and 12 are still buried in statistical noise**. Every conclusion quoted in the previous sections comes from those 18 only.

---

## 7. L4 — dynamics and transport (`l4_dynamics.py`, tier A)

**The question**: are the **atomic migration rates** the model produces right? Diffusion coefficients span orders of magnitude, so the comparison is made in log space.

### 7.1 Guards (before any number)

| metric_id | Unit | Meaning |
|---|---|---|
| `L4.equilibrium_segment@<state>.reference` / `.model` | bool | **Refuses a heating or cooling trajectory as an equilibrium diffusion reference.** A protocol whose temperature is a `(start, end)` pair fails directly; with per-frame temperatures, the mean-temperature drift between the two halves must be ≤ 100 K. The specification says the existing GST225 ramps cannot be used for equilibrium diffusion, so the rule lives in the module rather than in a notebook comment |
| `L4.unwrap_safety@<state>` | half of the shortest cell length | Maximum displacement between consecutive stored frames. XDATCAR stores **wrapped** fractional coordinates and unwrapping accumulates minimum-image steps; a value ≥ 0.5 means the sampling is too coarse, unwrapping is ambiguous and the MSD cannot be trusted |
| `L4.fit_window@<state>` | fs | The **absolute** time range of the fit window (by default 20%–80% of the lag span common to both sides). An absolute window rather than a fractional one stops a shorter model trajectory from quietly being fitted over a different time range |

### 7.2 Diffusion

MSD resolved by element and averaged over many time origins (the number of origins per lag is returned too, to show where the statistics run out). `D = slope/6` (3D), `1 Å²/fs = 0.1 cm²/s`.

| metric_id | Unit | Meaning |
|---|---|---|
| `L4.diffusion.<species>.D` | cm²/s | Self-diffusion coefficient (Ge/Sb/Te/all). `extra` carries the full fit information of both sides |
| `L4.diffusion.<species>.delta_log10D` | decades | `log10(D_model) − log10(D_ref)`. **The L4 scoring quantity**: log space treats over- and underestimation symmetrically, and 0.3 decades = a factor of 2 |
| `L4.diffusion_reference.<species>.D` | cm²/s | Reference value only, when there is no model trajectory |

**Three-way diffusive-regime test** (in `DiffusionFit`, reported with `failure_reason`): the **log–log slope** inside the window tells whether the diffusive regime has really been reached (ballistic → 2, caged → 0, diffusive → 1): `|slope−1| ≤ tol` → `diffusive`; `≤ 2·tol` → `marginal`; otherwise `not diffusive` (`failed`). The **relative spread of D over three sub-windows**, `d_window_spread`, is also reported: **that is the evidence that D has converged**.

> This run: slopes of 1.22 (AIMD) and 1.26 (MLIP), in the `marginal` band; the MLIP side has only 4.8 ps and barely reaches the diffusive regime; the sub-window spread is 7–9%, so the values themselves are converged to about 10%. **Quote the conclusion together with the window and the slope.**

---

## 8. L5 — process and pathway (`l5_pathway.py`, tier A)

**The question**: run the model through **exactly the same temperature programme as AIMD** (a melt–quench) and compare not only the end state but the structural evolution **along the way**, locating where the model **first** leaves the reference path.

### 8.1 Observables along the path (`PATHWAY_OBSERVABLES`)

40 sample points along programme progress 0→1; at each point 8 consecutive frames are averaged, and the in-window scatter is kept as that point's σ:

`mean_cn_all`, `mean_cn_{Ge,Sb,Te}`, `four_rings_per_atom`, `tetrahedral_fraction`, `rdf_first_peak_A`, `volume_per_atom_A3`, `potential_energy_per_atom_eV`.

### 8.2 Tolerance band: the reference's own fluctuation, not a magic number

`band = max(sigma_factor × in-window scatter of the reference, floor)`, with `sigma_factor = 3` by default. A fixed tolerance does not work here: the single-frame scatter of the Ge-Te RDF peak position (~0.25 Å), for example, is already several times larger than any physically meaningful shift. `PATHWAY_TOLERANCE_FLOORS` gives each observable a lower bound so that an accidentally small scatter cannot squeeze the band to zero.

| metric_id | Unit | Meaning |
|---|---|---|
| `L5.pathway.<obs>.first_divergence_progress` | fraction of the programme (0–1) | The **first** position where the model leaves the 3σ band for **2** consecutive sample points. Two consecutive points are required so a single noisy point is not counted as a divergence. `extra` carries `max_delta` and `max_delta_in_sigma_units`. **Use**: `failure_export` exports the **consecutive frames around the divergence** rather than only the final anomalous structure, because new data belongs where the divergence happens |
| `L5.pathway.<obs>.final_state_error` | units of the observable | End-state difference; `extra.within_reference_scatter` says whether it is still within the reference scatter |
| `L5.pathway_reference.<obs>.final` | | Reference end value, when there is no model trajectory |

### 8.3 Current limit on attribution (cite it with any L5 conclusion)

With a second MLIP quench under a different seed as a control, the ratio of the maximum deviation of `MLIP vs AIMD` to that of `MLIP(seed0) vs MLIP(seed1)` is **0.69–1.20**: every "divergence" this level flags **lies within the fluctuation the same model produces with another seed**. What makes L5 attributable is **independent reference replicas** (item 8 of TODO.md), not a longer single trajectory.

---

## 9. Cross-model ranking indicators (`ranking.py`)

**This is not grading.** Grading needs `sigma_eff`, which is still unavailable. This is a **relative comparison**: five models against one reference, one protocol, one set of frozen post-processing parameters and one code path, so their errors are **directly comparable** with each other even though none of them can get an absolute grade.

Scoring formula:

```
score_i = 100 × (x_best + floor) / (x_i + floor)
```

`floor` is the scale below which differences stop meaning anything physically: the measured statistical noise when there is one, otherwise an explicit physical resolution fixed in `INDICATORS`. Without a floor, an indicator on which every model is essentially exact would let numerical dust decide the ranking. Direction variants: `lower` (default, smaller is better), `higher`, `gap` (a safety-margin type, scored by the gap to the best rather than a ratio), `pass` (a 0–1 pass rate); `transform="one_minus"` turns a goodness of fit such as R² into an error `1−R²` before scoring.

30 indicators (with weights):

| Level | Indicator | Weight | floor | Note |
|---|---|---|---|---|
| L0 | force MAE | 2.0 | 0.02 eV/Å | Accuracy statement free of the energy zero |
| L0 | energy MAE | 1.5 | 5 meV/atom | After removing a constant offset per composition |
| L0 | stress RMSE | 1.0 | 0.1 meV/Å³ | Tier C; the Virial convention has only been audited |
| L0 | force 1−R² | 1.0 | 0.01 | Scored by unexplained variance |
| L0 | force MAE worst stage | 0.5 | 0.02 eV/Å | Worst of the five trajectory stages = transferability |
| L1 | L1 gate | 2.0 | – | Binary gate: says whether it passed, not by how much |
| L1 | PEC curves passing L1 | 1.5 | – | Fraction of the six curves that pass, finer than the gate |
| L1 | NVE drift | 2.0 | 0.05 meV/atom/ps | Tier A |
| L1 | energy-force consistency | 1.0 | 5e-3 | The floor is the framework's own pass threshold |
| L1 | PEC repulsive walls | 1.5 | – | Fraction of the six curves with a monotonic repulsive core |
| L1 | 2000 K min distance | 1.0 | 0.05 Å | `gap` type: larger is safer |
| L1 | lattice scan smoothness | 0.5 | 1.0 | |
| L1 | timestep drift spread | 1.0 | 0.05 | dt 3.0 vs 1.5 fs |
| L2 | relaxed volume error | 2.0 | 0.5 % | |
| L2 | relaxed cell-length error | 1.0 | 0.5 % | |
| L2 | EOS V0 error | 1.5 | 0.5 % | |
| L2 | EOS B0 error | 1.5 | 5 % | Curvature, much more sensitive than V0 |
| L2 | phase order | 1.5 | – | Only whether the sign is right |
| L3 | partial RDF W1 | 2.0 | 0.02 Å | Six pairs × two state points: the bond-length statement |
| L3 | ADF JS | 1.5 | 0.02 | |
| L3 | CN mean error | 2.0 | 0.05 | |
| L3 | homopolar bond error | 2.0 | 0.004 | The chemical-order signature |
| L3 | tetrahedral q W1 | 1.0 | 0.02 | Amorphous state only |
| L3 | ring distribution error | 1.5 | 0.02 rings/atom | Medium-range topology |
| L3 | ALTBC asymmetry error | 1.0 | 0.02 Å | Peierls distortion |
| L3 | principal peak error | 0.5 | 0.02 1/Å | |
| L4 | diffusion Δlog10D | 3.0 | 0.05 decades | The highest weight in the table |
| L4 | equilibrium segment | 1.0 | – | The model's own MD must first be an equilibrium segment |
| L5 | pathway endpoint error | 2.0 | 0.02 | |
| L5 | first divergence | 1.0 | 0.05 | `higher`: the further along the reference path, the better |

Level weights: `L0 0.20 / L1 0.15 / L2 0.15 / L3 0.25 / L4 0.15 / L5 0.10` (L3 is heaviest because the phase-one target is the structure of amorphous GST225; L1 is weighted as a gate rather than as a statement of quality).

**The L1 gate does not truncate the total score.** Rule 6 of the specification is about not spending compute on L4/L5 behind the gate, and by the time a ranking runs the compute has been spent. What a closed gate really means is that **the model's L2–L5 numbers are for reference only**: read the gate column before the ranking.

---

## 10. metric_id naming convention

```
L<level>.<family>.<object>.<field>@<state point>
```

- Level: `L0` `L1` `L2` `L3a` `L3b` `L4` `L5`
- `@<state point>`: specific to L3/L4/L5 (`amorphous_300K` / `liquid_2000K` / `melt_quench`), so numbers from different state points are never merged
- `<family>_reference.*`: **reference-side values only, when there is no model trajectory** (for example `L3a.rdf_reference.Ge-Te.r_peak`), kept apart from model–reference comparisons at the id level
- L0 binned metrics use `<family>.<tag>=<group>` (`L0.force_rmse.stage=1-2000K`)

---

## 11. Known limitations (read before citing any number)

1. **Every model–reference comparison is marked `provisional`**: each state point or phase has only one reference, so `sigma_ref` is unknown, and L0 also lacks `d_protocol` (the 24 SCFs under `results/bridge_set/`). The framework holds no formal grade.
2. **L0 stress is never scored**: the reference Virial protocol is not traceable and only its convention is audited.
3. **The L3b FSDP cannot be resolved on a 306-atom cell**; its three metrics are `skipped`, and the principal peak is used instead.
4. **Bond cutoffs for the L3 liquid need an explicit choice** (see 4.0), and `cutoff_source_state` must be cited with the conclusion.
5. **L4 has a single temperature point**, so no Arrhenius; the slope in the fit window falls in the `marginal` band.
6. **The L5 "first divergence" is not attributable yet** (see 8.3).
7. **The cubic-phase DFT EOS is still queued**: `L2.eos.cubic.*` is currently tier S (model-internal only); `run_l2.py` reads it automatically once it lands, with no code change.
8. The 306-atom reference has a number density of 0.02852 at/Å³, 8.2% below the project standard of 0.0309 at/Å³; any combined use with the phase-one melt–quench chains must say so.
