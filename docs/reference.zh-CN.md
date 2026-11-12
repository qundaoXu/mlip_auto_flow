# mlip_auto_flow 参考手册

[English](reference.md) · **中文** · [README](../README.zh-CN.md)

站点配置、完整的命令行参考、结果文件位置和仓库结构。项目功能与用法见 README。

## 配置

代码里不再写死任何站点路径。所有与机器有关的信息都放在一个配置文件 `mlip_auto_flow.yaml` 里（YAML 或 JSON 均可），评估端和主动学习端共用：

```bash
mlipeval config --init mlip_auto_flow.yaml   # 生成模板（与 config/mlip_auto_flow.example.yaml 相同）
vi mlip_auto_flow.yaml                       # 按本机填写
mlipeval config --check                      # 显示解析后的值，并检查每个路径是否存在
alflow config                                # 显示主动学习端会用到的设置
```

- **查找顺序：** `$MLIPEVAL_CONFIG` → 当前目录下的 `mlip_auto_flow.yaml` → 仓库根目录下的 `mlip_auto_flow.yaml`。三处都没有时，全部使用默认值。
- **优先级：** 环境变量 > 配置文件 > 默认值。任何标量都可以用 `MLIPEVAL_<SECTION>_<KEY>` 覆盖，例如 `MLIPEVAL_SCHEDULER_WALLTIME`。
- **路径写法：** 相对路径相对 `paths.workspace` 解析，支持 `${VAR}` 和 `~`。

默认值复现原来的工作区布局：参考数据和模型都放在 `eval_framework/` 的上一级目录。所以在原来的集群上，只要设置 `paths.workspace`、`environment.conda_root` 和 `scheduler.project` 三项即可。

### 需要配置的信息

**评估端（`mlipeval`）**

| 配置项 | 用途 | 默认值 | 环境变量 |
|---|---|---|---|
| `paths.workspace` | 参考数据与模型运行目录的根 | `eval_framework/` 的上一级 | `MLIPEVAL_WORKSPACE` |
| `paths.output` | `results/`、`data/`（缓存、参考标签、各模型输出）的写入位置 | `eval_framework/` | `MLIPEVAL_OUTPUT` |
| `paths.reference.gst_ref` | 旧版 AIMD、立方相种子结构、MP 结构 | `GST225_ref` | |
| `paths.reference.phase1` | phase-1 的 208 个 VASP 任务与 `task_manifest.csv` | `phase1_vasp_inputs_20260828` | |
| `paths.models.universal` | 通用势目录，每个模型一个子目录（checkpoint + `pair.in`） | `MLIP_run/universal_MLIP` | |
| `paths.models.specialised` | 专用势目录 | `MLIP_run/specialized` | |
| `paths.models.al_ladder` | AL 阶梯模型（`model/`、`pair/`） | `MLIP_run/specialized/al_ladder` | |
| `paths.models.gst_aging` | MACE-GST225-Aging 模型目录 | `../myproject/1-GST225-aging` | |
| `paths.training_sets.*` | 专用势训练集，只供泄漏筛查使用（GST-ACE-24、processed_NE、aging、AL 各代） | 见模板 | |
| `environment.conda_root` | conda 安装位置，作业脚本会 source 其中的 `etc/profile.d/conda.sh` | 由 `$CONDA_EXE` 推出 | `MLIPEVAL_CONDA_ROOT` |
| `environment.conda_envs` | conda 环境目录 | `<conda_root>/envs` | `MLIPEVAL_CONDA_ENVS` |
| `environment.backend_envs` | 后端 → 环境名（mace、nequip、deepmd、mattersim、chgnet） | mace / nequip / deepmd / gnnp / gnnp | |
| `lammps.executable` | L3–L5 MD 用的 LAMMPS 可执行文件 | `MLIP_run/universal_MLIP/_build/advancesoft_build/lmp` | `MLIPEVAL_LMP` |
| `lammps.common_dir` | mliap 共享脚本目录 | `MLIP_run/universal_MLIP/common` | |
| `lammps.al_ladder_export` | AL 阶梯模型导出 mliap 的脚本 | `MLIP_run/specialized/al_ladder/export_mliap.sh` | |
| `scheduler.project` | PBS `-P` 项目代码，不设置时省略这一行 | 无 | `MLIPEVAL_PBS_PROJECT` |
| `scheduler.queue` | PBS 队列，不设置时由 PBS 自动路由 | 无 | `MLIPEVAL_PBS_QUEUE` |
| `scheduler.select` / `walltime` | 资源请求与默认墙钟时间 | 1 GPU、12 CPU、225 GB；12 h | |
| `scheduler.modules` | 作业中执行 `module load` 的模块 | gcc/13.1.0、openmpi/gcc/64/4.1.8 | |
| `scheduler.require_compute_node` | 重计算是否必须在 PBS 作业内运行（工作站上设为 false） | true | |
| `models.<key>` | 新增或覆盖的模型条目，不必再改 `models.py` | 无 | |

每个 `models.<key>` 条目必填 `label`、`backend`、`checkpoint`，可选 `conda_env`、`dtype`、`native_protocol`、`lammps_route`、`lammps_pair_in`、`load_path`、`group`。L3–L5 需要 `lammps_route` 和 `lammps_pair_in`。

**运行时变量（不放进配置文件）：** `MLIPEVAL_MODEL`（当前模型）、`MLIPEVAL_MD_ENGINE`（`lammps` 或 `ase`）、`MLIPEVAL_THREADS`（线程数）。这三个由 `mlipeval run` 和 PBS 作业脚本自动设置。

**主动学习端（`alflow`，配置文件的 `alflow:` 段）**

| 配置项 | 用途 | 去向 |
|---|---|---|
| `alflow.run.*` | `activate_learning_full_flow.py` 的全部选项：数据路径、foundation model、抽样与划分比例、委员会规模、seed 与 epoch、MD 温度与步数、选帧阈值、轮数、DFT 目录与轮询设置 | 转成命令行参数 |
| `alflow.run.foundation_model` | 用来微调的 MACE foundation model | 也可以用 `$ALFLOW_FOUNDATION_MODEL` |
| `alflow.dft.partition` / `account` | Slurm 分区与账户 | `SBATCH_PARTITION` / `SBATCH_ACCOUNT` |
| `alflow.dft.modules` | VASP 运行前要 `module load` 的模块 | `VASP_MODULES` |
| `alflow.dft.vasp_cmd` | VASP 命令行 | `VASP_CMD`（默认 `mpirun -np $SLURM_NTASKS vasp_std`） |
| `alflow.dft.max_jobs` | 同时提交的 DFT 作业上限 | `ALFLOW_MAX_JOBS`（默认 30） |

VASP 模板（`al_flow/dft_input/std.run`）不含任何站点信息，只读取上面这些环境变量。POTCAR 需要自行放进 `dft_input/`。

## CLI

### `mlipeval` — 评估

```text
mlipeval config     [--check [--strict]] [--json] [--init PATH]   查看 / 检查 / 生成站点配置
mlipeval models     [--json]                              列出已登记模型及是否已有结果
mlipeval run LEVEL  --model KEY [driver 参数...]          LEVEL ∈ l0 l1 l2 md l3 l3u l4 l5 all
mlipeval submit     [--model KEY | --al-ladder] [...]     生成并提交 PBS 依赖链
mlipeval rank       [--models a,b,..] [--clean] [--markdown] [--out-stem S] [--level-weights L3=0.35,..]
mlipeval card       --model KEY [--against a,b,..] [--clean] [--json]   单模型报告卡（只读）
mlipeval candidates --model KEY [--out DIR]               查看 / 导出补算候选结构
mlipeval leakage    {h5,diagnostic,al-ladder}             训练集泄漏筛查
mlipeval report     --model KEY [--markdown]              phase-1 验收报告
mlipeval pack       reference [--out F] [--no-diagnostic]  冻结 L0–L2 的 DFT/AIMD 参考侧（在集群上做一次）
mlipeval pack       model --model KEY|--all [--out DIR]    把单个模型的 L0–L2 原始输出打成 .mlippack
mlipeval pack       verify PACK [--reference R]            对照参考包校验数据包，不做计算
mlipeval rescore    PACK [--levels L0,L1,L2] [--out DIR] [--compare]   在 CPU 上由数据包重算 L0–L2
mlipeval serve      [--host H] [--port P] [--store DIR]    本地 Workbench 后端：上传数据包、通过 HTTP 重算
```

`pack` / `rescore` / `serve` 把打分搬离集群：HPC 作业只负责调用模型，数据包携带模型输出，L0–L2 在打开数据包的机器上打分。格式、校验项和 API 见 [eval_framework/PACKS.md](../eval_framework/PACKS.md)。

`run` 会设置 `MLIPEVAL_MODEL=KEY`，再调用对应的 `run_*.py`，未识别的参数原样转发：

```bash
mlipeval run l0 --model nequip_oam_l --set audit --refresh
mlipeval run l3 --model mace_mpa0 --state canon_amorphous_300K
mlipeval run all --model mace_spec_gst                 # 整条链，L1 门控关闭时跳过 L4/L5
mlipeval submit --model mace_spec_gst --chain          # 每级一个 PBS 作业，按依赖串联
mlipeval rank --models mace_mpa0,nequip_oam_l,dpa4_pro,mattersim_5m,chgnet_030,mace_spec_gst,mace_spec_gst_ne,nequip_spec_gst,mace_gst_aging \
              --markdown --out-stem showcase9      # Fig. 1 的排名
mlipeval card --model mace_mpa0 --json                 # 机器可读报告卡
```

MACE 和 MD 计算必须在计算节点上运行，重计算的 driver 会自行检查。`models`、`rank`、`card`、`candidates`、`leakage` 是纯后处理，在登录节点上就能跑。

### `alflow` — 主动学习

```text
alflow [--config mlip_auto_flow.yaml] SUBCOMMAND ...     --config 也可以写在子命令之后
alflow config                                             显示将要使用的 run 选项与 DFT 环境变量
alflow run      [脚本参数...]                              整个循环（转发给 activate_learning_full_flow.py）
alflow train    --train --valid --test [--foundation-model M] [--n-comm 4] [--epochs 100] [--configs-only]
alflow sample   --models m1 m2 .. (--init-conf X | --traj T) [--temp 300] [--steps 1000] [--threshold 0.9] [--max-frames 200] [--device cuda]
alflow label    --frames F --template-dir D [--root dft_cal] [--submit] [--wait]
alflow collect  [--root dft_cal] [--out merged.extxyz] [--python-only]
alflow merge    --base train.xyz --new merged.extxyz --out next.xyz
```

```bash
alflow run --config mlip_auto_flow.yaml               # 读 alflow.run 与 alflow.dft 两段
alflow run --config examples/alflow.yaml               # 也接受只含 run 选项的扁平 YAML
```

分阶段运行时，每一步都输出一行 JSON，便于调度器记录状态。注意：`label` 会先删除 `--root` 下已有的数字编号子目录和 `*.xyz` 文件。

## 结果文件在哪里

| 内容 | 位置 |
|---|---|
| 每条指标（MetricResult） | `<paths.output>/results/models/<model>/L*.json` |
| 补算候选 | `<paths.output>/results/models/<model>/candidates/*.xyz` + `*_manifest.json` |
| 排名 | `<paths.output>/results/ranking_<stem>.json`、`RANKING_REPORT_<stem>.md` |
| PBS 日志 | `<paths.output>/results/pbs/*.log` |
| AL 每轮产物 | `committee*/`、`comm_traj/`、`dft_cal/`、`data/train_round*.xyz` |

## 仓库结构

```text
mlip_auto_flow/
├── eval_framework/          评估框架（scripts/、notebooks/、METRICS.md / METRICS.en.md）
├── al_flow/                 主动训练配套脚本
├── mlipeval_cli/            mlipeval / alflow 命令
├── config/mlip_auto_flow.example.yaml   站点配置模板（评估端 + 主动学习端）
├── examples/alflow.yaml     只含 AL run 选项的扁平配置示例
├── docs/                    项目主页（GitHub Pages）、reference.md
│   ├── index.html
│   ├── workbench/           MLIP Workbench（index.html、engine.js、examples/gst225/、使用说明）
│   ├── figures/             Fig. 1–8（PNG + PDF）
│   ├── notebooks/           每张图一本 notebook + style.py（Nature 版式，183 mm 通栏）
│   └── validation_methods_L0-L5.md / .en.md
├── README.md / README.zh-CN.md
└── pyproject.toml
```

主页上的图由 `docs/notebooks/fig*.ipynb` 从评估结果直接生成，需要先告诉它评估结果在哪里：

```bash
export MLIPEVAL_OUTPUT=/path/to/eval_framework   # 含 results/ 与 data/ 的工作副本（或在配置文件里设 paths.output）
cd docs/notebooks && for nb in fig*.ipynb; do jupyter nbconvert --to notebook --execute --inplace "$nb"; done
```
