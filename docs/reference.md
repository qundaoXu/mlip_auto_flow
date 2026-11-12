# mlip_auto_flow reference

**English** · [中文](reference.zh-CN.md) · [README](../README.md)

Site configuration, the full command-line reference, where results are written and the repository layout. The README covers what the project does and how to use it.

## Configuration

No site path is hard-coded. Everything machine-specific lives in one file, `mlip_auto_flow.yaml` (YAML or JSON), shared by the evaluation and the active-learning side:

```bash
mlipeval config --init mlip_auto_flow.yaml   # write the template (same as config/mlip_auto_flow.example.yaml)
vi mlip_auto_flow.yaml                       # fill in your machine
mlipeval config --check                      # resolved values, and whether every path exists
alflow config                                # what the active-learning side will use
```

- **Lookup order:** `$MLIPEVAL_CONFIG` → `mlip_auto_flow.yaml` in the current directory → `mlip_auto_flow.yaml` at the repository root. With none of them, every default applies.
- **Precedence:** environment variable > config file > default. Any scalar can be overridden as `MLIPEVAL_<SECTION>_<KEY>`, for example `MLIPEVAL_SCHEDULER_WALLTIME`.
- **Paths:** relative paths resolve against `paths.workspace`; `${VAR}` and `~` are expanded.

The defaults reproduce the original workspace layout, with reference data and models one level above `eval_framework/`. On the original cluster only three keys are needed: `paths.workspace`, `environment.conda_root` and `scheduler.project`.

### What needs to be configured

**Evaluation (`mlipeval`)**

| Key | Purpose | Default | Environment variable |
|---|---|---|---|
| `paths.workspace` | Root of the reference data and model runs | parent of `eval_framework/` | `MLIPEVAL_WORKSPACE` |
| `paths.output` | Where `results/` and `data/` (cache, reference labels, per-model outputs) are written | `eval_framework/` | `MLIPEVAL_OUTPUT` |
| `paths.reference.gst_ref` | Legacy AIMD, cubic seed structures, MP structures | `GST225_ref` | |
| `paths.reference.phase1` | The 208 phase-1 VASP tasks and `task_manifest.csv` | `phase1_vasp_inputs_20260828` | |
| `paths.models.universal` | Universal potentials, one subdirectory per model (checkpoint + `pair.in`) | `MLIP_run/universal_MLIP` | |
| `paths.models.specialised` | Specialised potentials | `MLIP_run/specialized` | |
| `paths.models.al_ladder` | AL-ladder models (`model/`, `pair/`) | `MLIP_run/specialized/al_ladder` | |
| `paths.models.gst_aging` | MACE-GST225-Aging model directory | `../myproject/1-GST225-aging` | |
| `paths.training_sets.*` | Training sets of the specialised potentials, read only by the leakage screens (GST-ACE-24, processed_NE, aging, AL generations) | see template | |
| `environment.conda_root` | conda installation; job scripts source its `etc/profile.d/conda.sh` | derived from `$CONDA_EXE` | `MLIPEVAL_CONDA_ROOT` |
| `environment.conda_envs` | conda environments directory | `<conda_root>/envs` | `MLIPEVAL_CONDA_ENVS` |
| `environment.backend_envs` | Backend → environment (mace, nequip, deepmd, mattersim, chgnet) | mace / nequip / deepmd / gnnp / gnnp | |
| `lammps.executable` | LAMMPS binary for L3–L5 MD | `MLIP_run/universal_MLIP/_build/advancesoft_build/lmp` | `MLIPEVAL_LMP` |
| `lammps.common_dir` | Shared mliap scripts | `MLIP_run/universal_MLIP/common` | |
| `lammps.al_ladder_export` | Script that exports AL-ladder models to mliap | `MLIP_run/specialized/al_ladder/export_mliap.sh` | |
| `scheduler.project` | PBS `-P` project code; the line is omitted when unset | none | `MLIPEVAL_PBS_PROJECT` |
| `scheduler.queue` | PBS queue; PBS routes the job when unset | none | `MLIPEVAL_PBS_QUEUE` |
| `scheduler.select` / `walltime` | Resource request and default walltime | 1 GPU, 12 CPU, 225 GB; 12 h | |
| `scheduler.modules` | Modules loaded in every job | gcc/13.1.0, openmpi/gcc/64/4.1.8 | |
| `scheduler.require_compute_node` | Whether heavy drivers must run inside a PBS job (set false on a workstation) | true | |
| `models.<key>` | Extra or overriding model entries, without editing `models.py` | none | |

Each `models.<key>` entry needs `label`, `backend` and `checkpoint`, and may set `conda_env`, `dtype`, `native_protocol`, `lammps_route`, `lammps_pair_in`, `load_path` and `group`. L3–L5 need `lammps_route` and `lammps_pair_in`.

**Runtime variables (not in the file):** `MLIPEVAL_MODEL` (current model), `MLIPEVAL_MD_ENGINE` (`lammps` or `ase`) and `MLIPEVAL_THREADS`. `mlipeval run` and the PBS job scripts set all three.

**Active learning (`alflow`, the `alflow:` section)**

| Key | Purpose | Passed as |
|---|---|---|
| `alflow.run.*` | Every option of `activate_learning_full_flow.py`: data path, foundation model, subsampling and split ratios, committee size, seeds and epochs, MD temperature and steps, selection threshold, rounds, DFT directories and polling | command-line arguments |
| `alflow.run.foundation_model` | MACE foundation model to fine-tune | or `$ALFLOW_FOUNDATION_MODEL` |
| `alflow.dft.partition` / `account` | Slurm partition and account | `SBATCH_PARTITION` / `SBATCH_ACCOUNT` |
| `alflow.dft.modules` | Modules to load before VASP | `VASP_MODULES` |
| `alflow.dft.vasp_cmd` | VASP command line | `VASP_CMD` (default `mpirun -np $SLURM_NTASKS vasp_std`) |
| `alflow.dft.max_jobs` | Limit on DFT jobs submitted at once | `ALFLOW_MAX_JOBS` (default 30) |

The VASP template (`al_flow/dft_input/std.run`) contains no site information and reads only these variables. Put your own POTCAR in `dft_input/`.

## CLI reference

### `mlipeval`: evaluation

```text
mlipeval config     [--check [--strict]] [--json] [--init PATH]   show / check / create the site config
mlipeval models     [--json]                              registered models and whether they have results
mlipeval run LEVEL  --model KEY [driver options...]       LEVEL ∈ l0 l1 l2 md l3 l3u l4 l5 all
mlipeval submit     [--model KEY | --al-ladder] [...]     write and submit a PBS dependency chain
mlipeval rank       [--models a,b,..] [--clean] [--markdown] [--out-stem S] [--level-weights L3=0.35,..]
mlipeval card       --model KEY [--against a,b,..] [--clean] [--json]   report card for one model (read-only)
mlipeval candidates --model KEY [--out DIR]               show / export candidate structures
mlipeval leakage    {h5,diagnostic,al-ladder}             training-set leakage screens
mlipeval report     --model KEY [--markdown]              phase-1 acceptance report
mlipeval pack       reference [--out F] [--no-diagnostic]  freeze the DFT/AIMD side of L0–L2 (once, on the cluster)
mlipeval pack       model --model KEY|--all [--out DIR]    collect one model's L0–L2 raw outputs into a .mlippack
mlipeval pack       verify PACK [--reference R]            check a pack against the reference, compute nothing
mlipeval rescore    PACK [--levels L0,L1,L2] [--out DIR] [--compare]   score L0–L2 from a pack on a CPU
mlipeval serve      [--host H] [--port P] [--store DIR]    local Workbench backend: upload packs, rescore over HTTP
```

`pack` / `rescore` / `serve` move the scoring off the cluster: the HPC job runs the model, the pack carries its outputs, and L0–L2 are scored wherever the pack is opened. Format, checks and API: [eval_framework/PACKS.md](../eval_framework/PACKS.md).

`run` sets `MLIPEVAL_MODEL=KEY` and calls the matching `run_*.py`; unknown options are passed through unchanged:

```bash
mlipeval run l0 --model nequip_oam_l --set audit --refresh
mlipeval run l3 --model mace_mpa0 --state canon_amorphous_300K
mlipeval run all --model mace_spec_gst                 # the whole chain; L4/L5 are skipped when the L1 gate is closed
mlipeval submit --model mace_spec_gst --chain          # one PBS job per level, chained by dependency
mlipeval rank --models mace_mpa0,nequip_oam_l,dpa4_pro,mattersim_5m,chgnet_030,mace_spec_gst,mace_spec_gst_ne,nequip_spec_gst,mace_gst_aging \
              --markdown --out-stem showcase9      # the ranking in Fig. 1
mlipeval card --model mace_mpa0 --json                 # machine-readable report card
```

MACE and MD work must run on compute nodes, and the heavy drivers check this themselves. `models`, `rank`, `card`, `candidates` and `leakage` are post-processing only and run on a login node.

### `alflow`: active learning

```text
alflow [--config mlip_auto_flow.yaml] SUBCOMMAND ...     --config may also follow the subcommand
alflow config                                             run options and DFT environment that will be used
alflow run      [script options...]                       the whole loop (forwarded to activate_learning_full_flow.py)
alflow train    --train --valid --test [--foundation-model M] [--n-comm 4] [--epochs 100] [--configs-only]
alflow sample   --models m1 m2 .. (--init-conf X | --traj T) [--temp 300] [--steps 1000] [--threshold 0.9] [--max-frames 200] [--device cuda]
alflow label    --frames F --template-dir D [--root dft_cal] [--submit] [--wait]
alflow collect  [--root dft_cal] [--out merged.extxyz] [--python-only]
alflow merge    --base train.xyz --new merged.extxyz --out next.xyz
```

```bash
alflow run --config mlip_auto_flow.yaml               # reads the alflow.run and alflow.dft sections
alflow run --config examples/alflow.yaml               # a flat YAML of run options also works
```

Each stage prints one line of JSON so a scheduler can track state. `label` first deletes the numbered subdirectories and `*.xyz` files under `--root`.

## Where the results are

| What | Where |
|---|---|
| Every metric (MetricResult) | `<paths.output>/results/models/<model>/L*.json` |
| Candidates for labelling | `<paths.output>/results/models/<model>/candidates/*.xyz` + `*_manifest.json` |
| Rankings | `<paths.output>/results/ranking_<stem>.json`, `RANKING_REPORT_<stem>.md` |
| PBS logs | `<paths.output>/results/pbs/*.log` |
| Per-round AL outputs | `committee*/`, `comm_traj/`, `dft_cal/`, `data/train_round*.xyz` |

## Repository layout

```text
mlip_auto_flow/
├── eval_framework/          evaluation framework (scripts/, notebooks/, METRICS.md / METRICS.en.md)
├── al_flow/                 active-learning scripts
├── mlipeval_cli/            the mlipeval / alflow commands
├── config/mlip_auto_flow.example.yaml   site-config template (evaluation + active learning)
├── examples/alflow.yaml     a flat config with AL run options only
├── docs/                    project page (GitHub Pages), reference.md
│   ├── index.html
│   ├── workbench/           MLIP Workbench (index.html, engine.js, examples/gst225/, guide)
│   ├── figures/             Fig. 1–8 (PNG + PDF)
│   ├── notebooks/           one notebook per figure + style.py (Nature layout, 183 mm wide)
│   └── validation_methods_L0-L5.md / .en.md
├── README.md / README.zh-CN.md
└── pyproject.toml
```

The figures are generated from the evaluation results by `docs/notebooks/fig*.ipynb`. Tell them where the results are first:

```bash
export MLIPEVAL_OUTPUT=/path/to/eval_framework   # working copy with results/ and data/ (or set paths.output)
cd docs/notebooks && for nb in fig*.ipynb; do jupyter nbconvert --to notebook --execute --inplace "$nb"; done
```
