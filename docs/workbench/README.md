# MLIP Workbench

[**Open the web showcase / 打开展示软件**](https://qundaoxu.github.io/mlip_auto_flow/workbench/)

MLIP Workbench is the interactive showcase for `mlip_auto_flow`. It brings the project's GST225 evaluation results, physical curves, model comparisons, trajectories and active-learning sampling into one browser application.

MLIP Workbench 是本项目的网页展示软件，将 GST225 评估结果、物理曲线、模型对比、轨迹和主动学习采样整合在一个交互界面中。

## Workspaces / 工作区

| Workspace / 工作区 | Interaction / 交互内容 |
| --- | --- |
| Report card / 评估报告卡 | Select a model, inspect scores and the L1 gate, and add diagnostic candidates to the basket / 选择模型，查看分数与 L1 门控，将诊断候选加入标注篮 |
| New evaluation / 新建评估 | Upload DFT reference and MLIP output level by level, analyse L0–L2 in the browser and generate a report card; one click fills in the files of any of the nine showcase models / 按层级上传 DFT 参考与 MLIP 产出，在浏览器中分析 L0–L2 并生成报告卡；可一键填入 9 个展示模型中任一个的文件 |
| Level detail / 分级详情 | Explore L0–L5 plots, overlay another model, switch states or element pairs, and filter metric records / 查看 L0–L5 图表，叠加模型，切换状态点或元素对，筛选指标 |
| Compare models / 模型对比 | Reweight the ranking, inspect score and indicator matrices, and replay active-learning generations / 调整排名权重，查看得分与指标矩阵，回放主动学习各代 |
| Trajectory lab / 轨迹分析 | Inspect energy, temperature and shortest contacts, select a time window and add it to the basket / 查看能量、温度和最小间距，选取时间片段并加入标注篮 |
| Sampling / 采样与标注 | Explore diversity and committee-uncertainty sampling, adjust selection settings and collect frames / 查看多样性及委员会不确定性采样，调整选帧设置并汇总帧 |

## Run locally / 本地运行

From the repository root / 在仓库根目录执行：

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory docs
```

Open [http://127.0.0.1:8765/workbench/](http://127.0.0.1:8765/workbench/). No installation or build step is needed. You can also open `index.html` directly in a browser; the built-in reference and the examples of *New evaluation* then cannot be loaded (browsers block reading files from a `file://` page), but your own uploads still work.

打开上述地址，无需安装依赖或构建；也可直接用浏览器打开 `index.html`，但此时「新建评估」无法读取内置参考和示例（浏览器禁止 `file://` 页面读取其他文件），自己上传的文件仍可使用。

## New evaluation / 新建评估

L0–L2 are computed in the browser by `engine.js`, a port of the `mlipeval` functions that produce those metrics: the L0 error statistics, the L1 checks and gate, and the L2 relaxation, phase-ladder, formation-energy and Birch–Murnaghan EOS metrics, plus the L0–L2 ranking indicators. Uploaded files are read locally and never leave the machine. With the built-in GST225 reference, the result is scored against the nine built-in models and can be added to the model selector as a report card. With an uploaded DFT reference, raw errors are reported without ranking. L3–L5 need AIMD and MLIP-MD trajectories and run on the local backend; in the demo they are taken from the example model's HPC results.

`examples/gst225/` holds the demo inputs in the upload formats: `reference/` (DFT labels of the 120 H5 audit frames, the L2 DFT relaxations and EOS, the AIMD NVE drift) and `models/<key>/` for the nine showcase models. `expected_metrics.json` holds the HPC values; for every example the page reports agreement (all L0–L2 metrics agree), and `tests/test_workbench_engine.py` checks it. The files are rebuilt with `MLIPEVAL_OUTPUT=… python docs/workbench/build_examples.py`. The formats are described in the panel under “What to upload”.

L0–L2 由 `engine.js` 在浏览器中计算，它逐个移植了 `mlipeval` 中产生这些指标的函数：L0 误差统计，L1 各项检查与门控，L2 的弛豫、相阶梯、形成能和 Birch–Murnaghan 状态方程，以及 L0–L2 的排名指标。上传的文件只在本机读取，不会上传到任何地方。使用内置 GST225 参考时，结果与 9 个内置模型一起计算相对分，可以生成报告卡加入模型选择器；使用自己上传的 DFT 参考时，只报告误差值，不参与排名。L3–L5 需要 AIMD 与 MLIP-MD 轨迹，在本地后端计算；演示中直接使用示例模型在 HPC 上的结果。

`examples/gst225/` 是按上传格式整理的演示数据：`reference/` 包括 120 帧 H5 审计集的 DFT 标签、L2 的 DFT 弛豫和状态方程、AIMD NVE 漂移；`models/<key>/` 是 9 个展示模型的产出。`expected_metrics.json` 记录 HPC 上的结果；每个示例分析完后，页面会显示与 HPC 是否一致（L0–L2 全部指标一致），`tests/test_workbench_engine.py` 也会检查这一点。这些文件由 `MLIPEVAL_OUTPUT=… python docs/workbench/build_examples.py` 重新生成。各文件的格式写在面板的「需要哪些数据」里。

## Data and execution / 数据与执行

`index.html` contains the demonstration data and the interface; `engine.js` the L0–L2 analysis. `mlipeval serve` (see [PACKS.md](../../eval_framework/PACKS.md)) serves this page at `/workbench/` together with an API that verifies uploaded evaluation packs and rescores L0–L2 on the CPU; Served this way, the page detects the backend and the report card's *Data in* panel accepts a `.mlippack`: it is uploaded, verified, rescored and added to the model selector under *Uploaded packs* (unranked, L0–L2 views). On GitHub Pages or from `file://` the page stays the static showcase. The embedded data is rebuilt from the evaluation outputs with `MLIPEVAL_OUTPUT=… python docs/workbench/build_data.py --inject docs/workbench/index.html`. Interactions and basket contents stay in browser memory and reset on reload. The showcase reads precomputed results; calculations, model training, data import/export and HPC submission use the repository's `mlipeval` / `alflow` workflow. Controls for unconnected operations are disabled.

`mlipeval serve`（见 [PACKS.md](../../eval_framework/PACKS.md)）在 `/workbench/` 提供本页，并提供校验评估数据包、在 CPU 上重算 L0–L2 的 API；以这种方式打开时，页面会检测到后端，报告卡的「数据接入」面板可以上传 `.mlippack`：上传、校验、重算后加入模型选择器的「上传的数据包」分组（不参与排名，可看 L0–L2）。在 GitHub Pages 或 `file://` 下仍是静态展示版。内嵌数据由 `MLIPEVAL_OUTPUT=… python docs/workbench/build_data.py --inject docs/workbench/index.html` 从评估输出重新生成。`index.html` 内嵌演示数据与界面，`engine.js` 是 L0–L2 的分析逻辑；交互状态和标注篮保存在浏览器内存中，刷新后重置。展示版读取预先计算的结果；计算、训练、数据导入导出与 HPC 提交使用仓库的 `mlipeval` / `alflow` 流程，尚未连接的操作在界面中禁用。

Plotly 2.35.3 loads from cdnjs, so the application needs network access for charts. All interface and chart fonts use local Arial, with the browser's fallback for unavailable glyphs. The palette follows the [project overview](https://qundaoxu.github.io/mlip_auto_flow/), with automatic light/dark themes and Chinese/English switching.

图表通过 cdnjs 加载 Plotly 2.35.3，需要联网。界面与图表统一使用本机 Arial，缺少的字形由浏览器回退。配色与项目介绍页一致，支持自动浅色/深色模式和中英文切换。

## Hosting / 发布

The repository's existing GitHub Pages configuration publishes `main:/docs`. This directory is served at `/mlip_auto_flow/workbench/`; the project overview and both repository READMEs link to it. Keep the relative project-overview link and the embedded evaluation data intact when editing the interface.

仓库现有 GitHub Pages 配置发布 `main:/docs`，本目录对应 `/mlip_auto_flow/workbench/`。项目介绍页与中英文 README 均提供入口。修改界面时保留相对导航链接及内嵌评估数据。
