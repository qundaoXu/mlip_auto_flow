# 第一阶段工程指南：MLIP L0–L5 分级诊断

**中文** · [English](validation_methods_L0-L5.en.md)

> 版本：v0.2，2026-08-28  
> 第一阶段对象：MACE-MPA-0 + GST225  
> 验证手段的背景、完整清单和文献案例见 `MLMD_AIMD_validation_summary.md`。

## 1. 第一阶段目标

第一阶段不是建立排行榜，也不自动重训势函数。目标是先跑通一个可复现的诊断闭环：

```text
读取参考数据和协议
→ 调用 MACE-MPA-0
→ 运行 L0–L5 分层任务
→ 输出数值、证据等级和失败构型
→ 给出训练数据缺口与补数据建议
```

第一阶段完成时应具备：

1. 一套统一的结构、协议和结果格式；
2. 一个固定版本的 MACE-MPA-0 适配器；
3. GST225 的 L0–L3 核心诊断、L4 单温度 smoke test 和 L5 melt–quench 路径测试；
4. 每项诊断都能回溯到模型、参考数据、协议和输入构型；
5. 报告能够指出失败环境，并导出建议补算的结构清单。

以下内容暂不进入第一阶段：模型微调、自动提交 DFT、公开总榜、成核率、界面晶化速度、熔点、玻璃化温度和器件尺度模拟。

## 2. 第一阶段输入

### 2.1 被测模型

使用官方 `medium-mpa-0` checkpoint。MACE-MPA-0 覆盖 89 种元素，训练数据为 MPTrj + sAlex，原生理论层级标记为 PBE+U；MACE 版本至少为 0.3.10。工程运行必须进一步固定：

- `mace-torch`、PyTorch、ASE 和 CUDA 版本；
- checkpoint 文件哈希；
- `default_dtype=float64`；
- `dispersion=False`，与当前无色散 PBE 参考保持一致；
- device、随机种子和确定性设置。

来源：[MACE 官方仓库](https://github.com/ACEsuit/mace)、[MACE foundation models](https://github.com/ACEsuit/mace-foundations)。

MACE-MPA-0 的训练理论与本项目 PBE 参考不是完全同一协议。第一阶段把它作为 zero-shot 被测模型，不把模型原生能量当作参考真值。

### 2.2 GST225 参考数据

本地数据位于 `GST225_ref/`。当前可直接使用的部分：

- 216 原子 AIMD 的 2000 K、2000→300 K、结构优化、300→200 K 和 300→400 K 轨迹；
- `AIMD/GST225_init_noStress.xyz`：615 个 216 原子抽样构型，含 `REF_energy`、
  `REF_forces` 和 `Virial`；覆盖三个晶胞尺寸及五类轨迹阶段；
- 一个 306 原子结构在 0.95–1.05 线性晶格缩放下的 11 个 PBE 单点。

详细审计与补算要求见 `GST225_reference_audit.md`。615 帧数据可支持 L0 energy/force
测试；Virial 需先核实符号、单位和来源协议。三个晶胞尺寸不能自动视为三个独立同状态轨迹，
因此当前数据仍不能给出可靠的参考方差。

## 3. 统一协议与结果

每个任务必须有唯一 `protocol_id`，至少记录：

- 结构 ID、组成、原子数、晶胞、初态和数据角色；
- 温度、压力、系综、时间步、步数、平衡/生产段、thermostat 和 seed；
- DFT code/version、functional、+U、dispersion、赝势、cutoff、k 点、smearing 和收敛阈值；
- 模型名称、checkpoint hash、dtype、device 和软件版本；
- 后处理参数：采样间隔、键截断、ring 算法、bin 宽和拟合窗口。

数据角色只允许以下三类：

- `training`：可用于训练；
- `diagnostic`：允许用于定位问题和选择补数据；
- `protected_audit`：不得用于补数据、微调或超参数选择。

结果至少包含：`metric_id`、value/unit、uncertainty、status、evidence tier、参考/模型协议、输入结构 ID、失败原因和输出文件。

## 4. 证据等级和判级

| 等级 | 含义 |
|---|---|
| A | 同协议、同尺寸 AIMD 直接对照 |
| B | 同一项目协议的静态 DFT/DFPT |
| C | 外部协议、已验证参考势或大尺度间接证据 |
| S | reference-free 自洽性与稳定性检查 |

正式判级只在参考统计充分时使用：

```text
sigma_ref = 独立参考轨迹之间的统计差异
d_protocol = 同一 bridge set 在两个参考协议下的差异
sigma_eff = max(sigma_ref, d_protocol / 2)
z_eff = distance(model, reference) / sigma_eff
```

`z_eff ≤ 1` 为 A，`1–2` 为 B，`2–4` 为 C，`>4` 为 F。只有一条参考轨迹时，不计算正式 grade，结果标为 `provisional`。

## 5. 分级诊断任务

### L0：数值参考一致性

**目的**：找出模型在目标构型上的能量、力和应力失效环境。

第一阶段实现：

1. 计算 energy/force/stress 的 MAE、RMSE、P99 和 max；
2. 能量按同组成去除常数偏移后比较；
3. 力误差按 Ge/Sb/Te、力幅、温度段和构型类型分箱；
4. 输出误差最大的帧和原子，并保留原结构；
5. 标记参考协议、模型原生协议和数据泄漏风险。

前置条件：先确认 `GST225_init_noStress.xyz` 的 DFT 协议和数据角色。energy/forces 可直接用于
L0；Virial 仅在核实符号、单位和分量顺序后使用。若协议无法追溯，选择 bridge set 按 canonical
协议重算，而不是立即重算全部 615 帧。

### L1：势能面健全性与轨迹稳定性

**目的**：在不依赖 DFT 参考的情况下，先排除明显病理。

第一阶段实现：

1. Ge–Ge、Ge–Sb、Ge–Te、Sb–Sb、Sb–Te、Te–Te 六种二体 PEC；
2. PEC 的连续性、短程排斥和 energy–force 数值导数一致性；
3. 平移、旋转、净力和应力数值导数检查；
4. 0.85–1.15 线性晶格缩放的不松弛能量扫描；
5. 2000 K 高温稳定性和最小原子间距分布；
6. 3 fs 与 1.5 fs 的时间步敏感性；
7. NVE 能量漂移和质心漂移。

出现 NaN/Inf、非物理短键、轨迹中断、能量/力不连续或显著不守恒时，L1 标记失败，暂不运行昂贵 L4/L5。

### L2：静态平衡性质

**目的**：检查基态、相对能量和 PES 曲率。

第一阶段实现：

1. cubic GST225 与 hexagonal GST225 的结构弛豫；
2. 两相的相对能量和排序；
3. 两相各自的 fixed-volume ionic-relaxed EOS；
4. 报告平衡体积和体模量。

当前 306 原子 11 点序列没有相名称且 `NSW=0`，只能作为不松弛体积扫描或测试 fixture，不能作为正式 relaxed EOS。

声子、弹性、缺陷和表面/界面性质留到第一阶段框架跑通之后。

### L3a：短程结构与化学序

**目的**：比较液体和非晶的局域结构。

第一阶段实现：

1. 六种 partial RDF 及峰位、峰高、峰宽和第一极小值；
2. 物种分辨 ADF；
3. Ge/Sb/Te 配位数完整分布；
4. 同极键比例；
5. Ge 四面体序参量分布；
6. 1-Wasserstein 和 Jensen–Shannon 距离。

每个 state point 的键截断从 AIMD partial RDF 第一极小值得到，然后对 AIMD 与 MACE-MD 同时冻结。

### L3b：中程结构与拓扑

**目的**：捕获 RDF/ADF 看不到、但会改变晶化的中程错误。

第一阶段实现：

1. ring-size distribution；
2. 四元环比例与平面度；
3. ALTBC；
4. S(q) 和 FSDP。

ring 定义、最大环尺寸和代码版本必须固定。第一阶段只选一种 ring 算法，不同时维护多种实现。

### L4：动力学

**目的**：确认结构正确是否同时对应合理原子运动。

第一阶段只实现一个固定温度液态 state point：

1. 分物种 MSD；
2. 自扩散系数；
3. 拟合窗口和该窗口的 log–log 斜率；
4. AIMD 与 MACE-MD 的 `log10(D)` 差。

现有升降温轨迹不能用于严格扩散对照。必须补一个固定温度、平衡后有生产段的 AIMD。多温度 Arrhenius 拟合留到下一步。

### L5：过程与相变

**目的**：比较给定温度程序下的路径，而不只比较终态。

第一阶段实现：

1. 对现有 2000 K→300 K 路径运行同尺寸 MACE-MD；
2. 比较 RDF 峰、CN、四元环、四面体 Ge 和体积随时间/温度的变化；
3. 比较最终非晶结构，不计算晶化速度或成核率。

现有流程只有一条轨迹，因此第一阶段结果标为 `provisional`。正式验证需增加独立 melt–quench replicas。

## 6. 从诊断到补数据建议

第一阶段不自动重训，但每类失败要产生可执行建议：

| 诊断 | 优先补充的结构 |
|---|---|
| 短程 PEC 或高温稳定性失败 | 六种元素对短程扫描、压缩结构、高温异常帧 |
| 某元素/力幅的 L0 尾部误差高 | 对应元素、配位和力幅的聚类代表帧 |
| 晶相排序或 EOS 错误 | cubic/hexagonal、应变结构和固定体积弛豫结构 |
| RDF/ADF/CN 错误 | 对应温度和密度的液体/非晶独立帧 |
| ring/ALTBC 错误 | 含关键四元环、长短键畸变和晶化前驱的结构 |
| 扩散错误 | 跃迁中间态、过冷液体和相邻温度的平衡帧 |
| melt–quench 路径错误 | 首次偏离参考前后的连续帧，而不是只取最终异常帧 |

导出的候选结构必须包含：失败来源、选择理由、去重/聚类信息、建议 DFT 协议和需要回归的 metric IDs。

## 7. GST225 第一阶段执行顺序

1. 整理 `GST225_ref` 元数据，同时读取原始 XDATCAR 与 615 帧带标签 extxyz；
2. 固定 MACE-MPA-0 checkpoint 和运行环境；
3. 先完成 L1，确认 MACE-MD 可稳定运行；
4. 对现有 DFT 11 点和 AIMD 路径运行 MACE，验证读取、计算和报告链；
5. 完成统一 DFT 收敛测试；建立 bridge set，核对现有 E/F/Virial 标签与 canonical 协议；
6. 完成 L0；
7. 完成 L3a/L3b 并用现有单轨迹得到 provisional 结果；
8. 补固定温度 AIMD 后完成 L4；
9. 补独立 melt–quench 后更新 L3/L5 的参考方差；
10. 输出失败构型和训练数据补充建议。

## 8. 第一阶段验收条件

- 同一协议可驱动 AIMD 参考读取和 MACE-MD；
- 所有结果都能回溯到 structure/protocol/model ID；
- L0 能定位到具体异常原子和构型；
- L1 的故障会阻止无意义的高成本任务；
- L3 使用完全相同的 cutoff、bin 和 ring 参数；
- L4 报告拟合窗口，不接受升降温轨迹伪装成平衡扩散；
- 单轨迹结果明确标记 provisional；
- 报告区分模型误差、参考统计噪声和协议差异；
- 能导出一份去重后的候选 DFT 补算结构列表；
- 不读取或修改 protected audit set 来指导补数据。

## 9. 第一阶段工程清单

- [ ] 定义 structure、protocol、model 和 result 的最小 schema；
- [ ] 编写 VASP/XDATCAR 读取与 extxyz 转换；
- [ ] 固定并校验 MACE-MPA-0 checkpoint；
- [ ] 实现统一 calculator adapter；
- [ ] 实现 L0 和 L1；
- [ ] 实现 L2 的结构弛豫与 EOS；
- [ ] 实现 GST225 的 L3a/L3b；
- [ ] 实现单温度 L4；
- [ ] 实现现有 melt–quench 的 L5 路径比较；
- [ ] 实现独立轨迹不确定度和 protocol mismatch 标记；
- [ ] 实现失败构型导出；
- [ ] 生成 GST225 第一版诊断报告。
