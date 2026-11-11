# MLIP Workbench

[**Open the web showcase / 打开展示软件**](https://qundaoxu.github.io/mlip_auto_flow/workbench/)

MLIP Workbench is the interactive showcase for `mlip_auto_flow`. It brings the project's GST225 evaluation results, physical curves, model comparisons, trajectories and active-learning sampling into one browser application.

MLIP Workbench 是本项目的网页展示软件，将 GST225 评估结果、物理曲线、模型对比、轨迹和主动学习采样整合在一个交互界面中。

## Workspaces / 工作区

| Workspace / 工作区 | Interaction / 交互内容 |
| --- | --- |
| Report card / 评估报告卡 | Select a model, inspect scores and the L1 gate, and add diagnostic candidates to the basket / 选择模型，查看分数与 L1 门控，将诊断候选加入标注篮 |
| Level detail / 分级详情 | Explore L0–L5 plots, overlay another model, switch states or element pairs, and filter metric records / 查看 L0–L5 图表，叠加模型，切换状态点或元素对，筛选指标 |
| Compare models / 模型对比 | Reweight the ranking, inspect score and indicator matrices, and replay active-learning generations / 调整排名权重，查看得分与指标矩阵，回放主动学习各代 |
| Trajectory lab / 轨迹分析 | Inspect energy, temperature and shortest contacts, select a time window and add it to the basket / 查看能量、温度和最小间距，选取时间片段并加入标注篮 |
| Sampling / 采样与标注 | Explore diversity and committee-uncertainty sampling, adjust selection settings and collect frames / 查看多样性及委员会不确定性采样，调整选帧设置并汇总帧 |

## Run locally / 本地运行

From the repository root / 在仓库根目录执行：

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory docs
```

Open [http://127.0.0.1:8765/workbench/](http://127.0.0.1:8765/workbench/). No installation or build step is needed. You can also open `index.html` directly in a browser.

打开上述地址，无需安装依赖或构建；也可直接用浏览器打开 `index.html`。

## Data and execution / 数据与执行

The standalone HTML contains the demonstration data and application logic. Interactions and basket contents stay in browser memory and reset on reload. The showcase reads precomputed results; calculations, model training, data import/export and HPC submission use the repository's `mlipeval` / `alflow` workflow. Controls for unconnected operations are disabled.

独立 HTML 内嵌演示数据与交互逻辑，交互状态和标注篮保存在浏览器内存中，刷新后重置。展示版读取预先计算的结果；计算、训练、数据导入导出与 HPC 提交使用仓库的 `mlipeval` / `alflow` 流程，尚未连接的操作在界面中禁用。

Plotly 2.35.3 loads from cdnjs, so the application needs network access for charts. All interface and chart fonts use local Arial, with the browser's fallback for unavailable glyphs. The palette follows the [project overview](https://qundaoxu.github.io/mlip_auto_flow/), with automatic light/dark themes and Chinese/English switching.

图表通过 cdnjs 加载 Plotly 2.35.3，需要联网。界面与图表统一使用本机 Arial，缺少的字形由浏览器回退。配色与项目介绍页一致，支持自动浅色/深色模式和中英文切换。

## Hosting / 发布

The repository's existing GitHub Pages configuration publishes `main:/docs`. This directory is served at `/mlip_auto_flow/workbench/`; the project overview and both repository READMEs link to it. Keep the relative project-overview link and the embedded evaluation data intact when editing the interface.

仓库现有 GitHub Pages 配置发布 `main:/docs`，本目录对应 `/mlip_auto_flow/workbench/`。项目介绍页与中英文 README 均提供入口。修改界面时保留相对导航链接及内嵌评估数据。
