# eval 框架指标说明（L0–L5）

本文件说明 `eval/scripts/mlipeval/` **当前已实现**的每一个指标：它由哪段代码产出、
数学定义是什么、物理上在问什么问题、以及什么情况下它会被判为 `failed` / `skipped`。

规范源头是 `validation_methods_L0-L5.md`（任务定义与验收条件）与 `database_link.md`
（协议隔离与数据角色）；本文件只描述**代码里真正算出来的东西**，不复述规范里尚未落地的部分。

结果文件位置：`eval/results/*.json`（第一阶段 MACE-MPA-0）与
`eval/results/models/<model_key>/*.json`（五模型扩展）。

| 级别 | 产出文件 | 指标条数（本轮实测） | 主问题 |
|---|---|---|---|
| L0 | `L0_<protocol_id>.json` | 34 | 势函数在**给定构型**上把能量/力/应力算得准不准 |
| L1 | `L1_<model>.json` | 23 | 势能面本身是否**自洽、连续、可积分**（无参考、tier S） |
| L2 | `L2_<model>.json` | 13 | 有序相的**静态平衡性质**：结构、相序、状态方程 |
| L3a | `L3a_<state>.json` | 52 ×2 态 | 无序相的**短程结构与化学序** |
| L3b | `L3b_<state>.json` | 18 ×2 态 | 无序相的**中程序与拓扑** |
| L3-u | `L3_uncertainty_<state>.json` | 33 ×2 态 | 给每个 L3 标量配**可测误差尺度**并判可分辨性 |
| L4 | `L4_<state>.json` | 12 | **输运**：MSD 与自扩散系数 |
| L5 | `L5_<programme>.json` | 16 | **过程/路径**：熔淬程序沿程复现与首次偏离 |
| 跨模型 | `ranking.json` | 30 项打分 | 同一参考下五个模型的**相对**排名 |

---

## 0. 所有级别共用的记账约定

### 0.1 一条指标记录（`schema.MetricResult`）

每条指标至少带：`metric_id`、`level`、`value`、`unit`、`status`、`evidence_tier`、
`grade`/`z_eff`、`reference_protocol`、`model_protocol`、`structure_ids`、
`failure_reason`、`outputs`、`extra`。这是"可回溯"要求（规范第 3 节）的代码化：
任何一个数都能回答"谁跟谁比、用什么协议、错在哪个构型上"。

### 0.2 `status`：这个数处于什么状态

| status | 含义 |
|---|---|
| `ok` | 算出来了，且这是一个自洽/无参考的判定（或本身就是描述量） |
| `provisional` | 算出来了，但**参考散布 `sigma_ref` 未知**，因此不给正式等级 |
| `failed` | 模型或该次运行没通过这项检查 |
| `skipped` | 前置条件不满足（如 L1 门控关闭、FSDP 未被盒子分辨） |
| `error` | 分析本身没跑起来（例如某后端无法计算二原子体系） |

**为什么绝大多数模型-参考对比都是 `provisional`**：每个状态点/相目前只有一条参考轨迹，
`sigma_ref` 无从测量；L0 还缺 `d_protocol`（需要 bridge set 的 24 个 SCF）。
`schema.grade_metric` 在 `n_independent_references < 2` 时**拒绝编造等级**。

### 0.3 `evidence_tier`：这个比较有多硬

| tier | 含义 | 典型指标 |
|---|---|---|
| **A** | 同协议、同尺寸的 AIMD 直接对照 | L3/L4/L5 全部、L1 的 NVE 漂移（有 AIMD 前飞对照时） |
| **B** | 同项目协议下的静态 DFT/DFPT | L0 的 E/F、L2 的体积/EOS/相序 |
| **C** | 外部协议 / 间接证据 | L0 的 stress（参考 Virial 约定只做过审计，未重新推导） |
| **S** | 无参考的自洽性与稳定性检查 | 全部 L1、L0 的健康检查、L4 的平衡性/解缠绕守卫 |

### 0.4 判级公式（尚未激活）

```
sigma_eff = max(sigma_ref, d_protocol / 2)
z_eff     = |distance(model, ref)| / sigma_eff
z ≤ 1 → A;  1–2 → B;  2–4 → C;  > 4 → F
```
`sigma_ref` = 参考侧独立复现之间的散布；`d_protocol` = 两套 DFT 协议在同一批结构上的系统差。
两者任一缺失即 `provisional`。当前二者都缺，所以**框架内没有任何一条正式 grade**。

### 0.5 协议隔离与能量偏移

- 每套标签有自己的 `protocol_id`，不同协议的数据**不允许并进同一个 RMSE**。
- 能量比较前**只允许**在"同组成 + 同协议"内扣除一个每原子常数偏移
  （`l0_reference.constant_energy_offset`）；力与应力**不做任何经验校正**。

---

## 1. L0 — 参考一致性（`l0_reference.py`）

**在问什么**：给定 DFT 已经标好的 615 帧构型，模型在**同样的构型**上给出的
能量、力、应力偏离多少。这是唯一不涉及模型自己跑动力学的一级，所以它的误差
是"纯拟合误差"，不含轨迹发散的贡献。

### 1.1 模型健康

| metric_id | 单位 | 定义与物理含义 |
|---|---|---|
| `L0.model_eval_failures` | count | 计算器直接抛异常的帧数。非零即 `failed`——模型在参考构型上都跑不动，后面所有数都不必看 |
| `L0.nonfinite_energies` | count | 能量为 NaN/Inf 的帧数。同上 |

### 1.2 能量（tier B）

先按组成拟合每原子常数偏移 `offset_c = Σ(E_model − E_ref) / Σ N`，再比较
`ΔE/N = (E_model − N·offset − E_ref)/N`。

| metric_id | 单位 | 含义 |
|---|---|---|
| `L0.energy_mae` / `_rmse` / `_p99` / `_max` | meV/atom | 每原子能量误差的平均/均方根/99 分位/最大值。RMSE 远大于 MAE 说明误差由少数帧主导；P99 与 max 是"最坏情况"，决定 MD 里会不会出现局部虚假驱动力 |
| `L0.energy_offset_spread` | meV/atom | 不同组成拟合出的 offset 的极差。**这是一个协议诊断量**：若各组成需要差别很大的偏移，说明模型与参考不只差一个常数（元素参考态/赝势层面的差异），能量比较的前提被破坏。本轮为 0（单一组成） |

> 物理提醒：能量 MAE 的绝对值不可跨协议解读。本轮 14.7 meV/atom 是 MACE-MPA-0 的
> 训练协议（PBE+U, MPtrj/sAlex）与 GST225 legacy 静态协议之间的差 + 模型误差之和，
> 二者未分离——分离它们正是 bridge set 要做的事。

### 1.3 力（tier B）

| metric_id | 单位 | 含义 |
|---|---|---|
| `L0.force_mae` / `_rmse` / `_p99` / `_max` | eV/Å | 力**分量**误差统计。力不需要能量零点，因此是最干净的精度陈述，跨协议解读也最安全 |
| `L0.force_component_r2` | – | 参考力分量与模型力分量的 Pearson R²。它衡量"方向与量级的整体相关性"；R² 高而 MAE 大 = 系统性缩放，R² 低 = 结构性错误 |
| `L0.force_mae.species=<X>` | eV/Å | 按元素分箱。哪个元素的化学环境没学好（GST 里 Ge 的四面体/八面体二态最容易出问题） |
| `L0.force_vector_mae.\|F\|=[a,b)` | eV/Å | 按**参考力大小**分箱的力矢量误差。低 \|F\| 桶反映近平衡区（决定声子与弛豫），高 \|F\| 桶反映强排斥区（决定高温 MD 的稳定性）。误差随 \|F\| 单调上升是正常的；若低 \|F\| 桶误差反而大，说明平衡附近的势能面被抹平 |
| `L0.force_rmse.<tag>=<group>` | eV/Å | 按任意帧级标签分箱的逐帧力 RMSE。本轮用了 `stage`（1-2000K / 2-300K / 3-opt / 4-MD / 5-MD）、`state_kind`（liquid / quench / amorphous_ramp / relaxed）、`density_tag`（d27/d28/d29）。**这是迁移性指标**：某一相/某一密度显著变差，就是补数据的靶点 |

### 1.4 应力（tier C）

| metric_id | 单位 | 含义 |
|---|---|---|
| `L0.stress_rmse` | meV/Å³ | 应力张量分量的 RMSE。参考侧的 `Virial` 符号/单位/分量顺序**未经协议溯源**，代码用 `io_labeled.audit_virial_convention` 与模型应力做相关性审计，本轮定出 `stress = −Virial/V`（r = 0.986，反号 −0.986）。审计不能定论时该项 `skipped`。**永远只作诊断，不参与打分** |

### 1.5 最坏构型导出

| metric_id | 含义 |
|---|---|
| `L0.worst_cases` | 力误差最大的 10 帧与 25 个原子（含元素、帧号、误差值），结构一并保留。用于 `failure_export` 生成补算候选；`protected_audit` 角色的帧会被拒绝导出 |

---

## 2. L1 — 势能面自洽性与轨迹稳定性（`l1_pes.py`，全部 tier S）

**在问什么**：不看参考，只看模型自己。这条势能面是不是一个**合法的**势能面：
连续、可微、力=−∇E、有排斥核、能积分出不发散的轨迹。
L1 是**门控级**：`l1_gate()` 有任一 `failed` 就不许花机时跑 L4/L5（规范第 6 条）。

### 2.0 噪声地板（本框架相对规范的修正）

`energy_noise()` 用三种**严格不变**的变换（刚体平移、原子序号置换、二者叠加）实测
模型在该结构上的能量噪声。float32 模型在 306 原子胞上噪声约 1e-4 eV，用 float64 时代的
1 µeV 绝对阈值会把舍入误差判成对称性破缺。因此所有不变性/有限差分检查的容差都取
`max(绝对阈值, 5 × 实测噪声传播值)`，并把实际容差写进 `extra`。
**置换探针是必需的**：从分数坐标建图的模型（CHGNet）对纯平移给出精确 0，会谎报零噪声。

### 2.1 二体势能曲线 PEC

`L1.pec.<A>-<B>`（六条：Ge-Ge/Ge-Sb/Ge-Te/Sb-Sb/Sb-Te/Te-Te），
value = `{r_min, e_min}`，即键的平衡距离与阱深。真空盒中两原子，r 从 1.0 到 8.0 Å，步长 0.02 Å。

判定（任一触发即 `failed`，写进 `failure_reason`）：

| 检查 | 阈值 | 物理含义 |
|---|---|---|
| 短程排斥单调 | r ≤ 1.6 Å 内 E(r) 必须单调上升 | 没有排斥核的势会在高温 MD 里让原子"穿透"，是灾难性失效 |
| 曲线上无 NaN/Inf | — | 外推区数值崩溃 |
| 能量/力导数一致 | `max\|F_analytic − (−dE/dr)\|/max\|F\| ≤ 0.05` | 力必须真的是能量的梯度 |
| 导数残差 O(h²) 收敛 | `residual(2h)/residual(h) ≥ 2.5` | **无量纲连续性判据**。中心差分是 O(h²)，光滑曲线该给 ~4；不连续点的残差不随 h 缩小，比值趋近 1。仅当残差高于该曲线自身噪声地板时才生效，否则标注为无信息 |

> 为什么不用"能量跳变绝对阈值"：2 Å 处 Te-Te 仍在排斥壁上，相邻 0.02 Å 采样点间
> 合法地相差 ~0.5 eV。`max_energy_jump_eV` 只作诊断记录。

### 2.2 对称性与守恒

| metric_id | 单位 | 定义 | 物理含义 |
|---|---|---|---|
| `L1.translation_invariance` | eV | 刚体平移前后能量差 | 平移不变性；破缺意味着体系有"绝对位置"，动量不守恒 |
| `L1.rotation_invariance` | eV | 非周期团簇转 37° 前后能量差 | 转动不变性（周期盒本身破坏转动对称，故在团簇上测）；破缺意味着角动量不守恒 |
| `L1.net_force` | eV/Å/atom | `\|Σ_i F_i\|/N` | 牛顿第三定律 / 平移不变性的导数形式。非零净力会让质心自发加速 |
| `L1.net_torque` | eV/atom | `\|Σ_i r_i × F_i\|/N`（相对质心） | 周期体系下净力矩不为零是正常的（仅记录，不判定） |
| `L1.force_vs_dE_dx` | – | 随机 8 个原子 × 3 方向，`max\|F_analytic − F_numeric\|/max\|F\|` | 力是否等于 −∂E/∂x。差分步长按实测噪声取 σ^(1/3) 最优标定 |
| `L1.stress_vs_dE_dstrain` | – | 六个 Voigt 分量，`(1/V)dE/dε` 中心差分对照解析应力 | 应力是否等于能量对应变的导数。**用框架自写的差分**（`_numerical_stress_voigt`），因为 ASE 的实现要 `free_energy`，CHGNet 没有——只对五分之四模型生效的检查比没有检查更糟 |

### 2.3 晶格标度扫描（未弛豫）

各向同性线性缩放 0.85–1.15，共 31 点。

| metric_id | 单位 | 含义 |
|---|---|---|
| `L1.lattice_scan_finite` | count | 31 点中成功算出的点数；少一个即 `failed`（外推区崩溃） |
| `L1.lattice_scan_smoothness` | max/median \|d²E\| | E(scale) 二阶差分的**离群比**，> 25 判失败。用比值而非绝对值：压缩区本来就强弯曲，真正的扭结表现为单点离群 |
| `L1.lattice_scan_compression_monotonic` | bool | 最压缩的三分之一区间内 E 必须单调（向压缩方向单调上升）。压缩不涨能量 = 势能面在高密度侧有假极小，MD 会塌缩 |

### 2.4 高温稳定性

| metric_id | 单位 | 含义 |
|---|---|---|
| `L1.stability_2000K.completed` | bool | 2000 K NVT 400 步是否跑完（不 NaN、不炸） |
| `L1.stability_2000K.min_distance` | Å | 全程**最近原子间距**。< 1.6 Å 判失败：稳定的 GST 熔体里不该出现这么近的接触，出现即排斥核不足 |
| `L1.stability_2000K.mean_temperature` | K | 恒温器是否真的把温度控在目标值（诊断） |

### 2.5 时间步敏感性与 NVE 漂移

| metric_id | 单位 | 含义 |
|---|---|---|
| `L1.timestep_sensitivity.drift_spread` | meV/atom/ps | 同样物理时长下 dt = 3.0 fs 与 1.5 fs 的 NVE 漂移之差。> 5 判失败。**这是项目 3.0 fs AIMD 时间步的合法性检验**：若漂移强烈依赖 dt，说明势能面有高频结构，3 fs 积分不成立 |
| `L1.timestep_sensitivity.completed.dt=<x>fs` | bool | 各时间步的运行是否完成 |
| `L1.nve_drift.dt=3fs` | meV/atom/ps | NVE 段内总能对时间的线性漂移率。\|drift\| > 5 判失败。**能量守恒是可积分性的直接证据**；有 AIMD 前飞对照时升级为 tier A（同胞、同密度、同温度、同 dt） |
| `L1.com_drift.dt=3fs` | Å | 质心位移最大值，> 1e-3 判失败。净力守恒的动力学后果 |

### 2.6 门控

`l1_gate()` 返回 `{pass, n_checks, n_failed, failed_metrics, decision}`，
写进 L1 结果文件的 `context`。`run_all.py` 在未通过时停在 L2 之前。

---

## 3. L2 — 静态平衡性质（`l2_static.py`，tier B/S）

**在问什么**：模型对**有序相**（立方与六方 GST225）的平衡结构、相对稳定性、
体积-能量关系是否正确。这是连接 L0（单点精度）与 L3+（无序动力学）的桥梁：
一个单点很准但弛豫后结构跑偏的模型，说明误差在"梯度方向"上是系统性的。

### 3.1 弛豫（`FrechetCellFilter` + FIRE，fmax = 0.01 eV/Å）

| metric_id | 单位 | 含义 |
|---|---|---|
| `L2.relax.<label>.energy_per_atom` | eV/atom | 弛豫后每原子能量（tier S，模型内部量）。未收敛到 fmax 即 `failed` |
| `L2.relax.<label>.volume_error` | % | 相对同一初始结构的 DFT 弛豫体积的偏差。**过结合/欠结合的最直接读数**：负值 = 体积偏小 = 键偏强 |
| `L2.relax.<label>.cell_length_error` | % | 三个晶胞边长误差的最大值（逐轴值在 `extra`）。与体积误差分开看能区分"整体缩放"与"各向异性畸变"——六方相的 c 轴（含 van der Waals 型间隙）是常见失效点 |

### 3.2 相序

| metric_id | 单位 | 含义 |
|---|---|---|
| `L2.phase_order.<phase>_minus_<base>` | meV/atom | 两相每原子能量差（同组成、同协议，故常数偏移自动抵消）。`extra` 里带 DFT 的对应值与误差；**符号与 DFT 相反直接判 `failed`**——相序搞反的势函数不可能给出正确的结晶行为。本轮 `cubic_minus_hexagonal = +44.9 meV/atom`（六方更稳，符合预期） |

### 3.3 状态方程（EOS）

固定体积 + **离子弛豫**的体积扫描（线性标度 0.92–1.08，与 phase-1 的 `H2_*_v092..v108` 一致），
三阶 Birch-Murnaghan 拟合。

`load_dft_eos` **拒绝读 `NSW ≤ 0` 的任务**：未弛豫的体积扫描不是 EOS
（GST225_ref 里那套 306 原子 11 点 `NSW=0` 扫描因此不被接受）。

| metric_id | 单位 | 含义 |
|---|---|---|
| `L2.eos.<label>.V0` | Å³/atom | 拟合的平衡体积。与弛豫体积互为独立估计 |
| `L2.eos.<label>.B0` | GPa | 体弹模量 = E(V) 曲线在 V0 处的**曲率**。它比 V0 敏感得多：体积对得上而 B0 差 20% 说明势阱的宽度（即声子硬度、热膨胀）不对 |
| `L2.eos.<label>.V0_error` / `B0_error` | % | 相对 DFT 拟合值的相对误差。`extra` 里同时带两侧的完整拟合参数（含 `Bp` 与拟合 RMSE，后者是"这条曲线值不值得拟合"的自检） |

---

## 4. L3a — 无序相短程结构与化学序（`l3a_short_range.py`，tier A）

**在问什么**：非晶/液态 GST225 里，**键长对不对、配位对不对、化学序对不对**。
第一阶段的核心结论就出自这一级。

### 4.0 参数冻结（这一级的前提）

`freeze_params_from_reference()` 从**参考（AIMD）**的 partial RDF 第一极小值导出每对元素的
键截断，打包成 `StructureParams`，**原封不动**用于 MLIP 轨迹。
`cutoff_diagnostics` 逐对记录峰位、极小位、极小深度 `(g_peak−g_min)/g_peak`、
以及该截断是实测还是从总 RDF 继承。

> **使用时必须做的显式选择**：300 K 非晶上 6 对里 5 对有良定义极小（depth 0.80–0.88），工作良好；
> 2000 K 液态上 4 对退化，回退到总 RDF 的 4.49 Å 会越过第一壳层，把 `<CN>` 抬到 9.8（数进第二壳层）。
> 用 `--cutoffs-from amorphous_300K` 继承后液态 `<CN>` 回到 4.34。
> 这个选择写在结果 JSON 的 `cutoff_source_state`，**必须随结论一起引用**。

### 4.1 分布距离的两把尺子

两者对每个分布都算，因为它们对不同错误敏感：

- **1-Wasserstein `W1`**（"搬土距离"）：对分布的**位移**敏感，量纲同横轴（Å / deg）。
  峰整体移了 0.1 Å，W1 ≈ 0.1。
- **Jensen-Shannon 距离**（base 2，取值 [0,1]）：对分布的**形状/权重重分配**敏感，无量纲。
  峰位不动但变高变窄，W1 几乎不变而 JS 上升。

### 4.2 偏径向分布函数 partial RDF

六对 `Ge-Ge, Ge-Sb, Ge-Te, Sb-Sb, Sb-Te, Te-Te`：

| metric_id | 单位 | 含义 |
|---|---|---|
| `L3a.rdf.<pair>.r_peak` | Å | 第一峰位 = **该对元素的典型键长** |
| `L3a.rdf.<pair>.g_peak` | – | 第一峰高 = 该键的**相对丰度/有序度**。GST 里 `Ge-Ge`、`Sb-Sb` 的峰高直接就是同极键（错键）含量的读数 |
| `L3a.rdf.<pair>.fwhm` | Å | 第一峰半高全宽 = 键长分布的**宽度**，反映局域无序度与热涨落 |
| `L3a.rdf.<pair>.wasserstein1` | Å | 整条 g(r) 的位移型差异 |
| `L3a.rdf.<pair>.jensen_shannon` | – | 整条 g(r) 的形状型差异 |
| `L3a.rdf_reference.<pair>.<field>` | | 无模型轨迹时只输出参考值（含 `first_minimum`，即所用截断） |

每条 `extra` 里都带 `reference` / `model` / `difference` 三元组。

### 4.3 键角分布 ADF

按中心原子元素（Ge / Sb / Te / all）分解，归一化到单位面积。

| metric_id | 单位 | 含义 |
|---|---|---|
| `L3a.adf.<species>.jensen_shannon` | – | 角分布形状差异 |
| `L3a.adf.<species>.wasserstein1` | deg | 角分布位移差异 |

物理上：**90° 附近的峰 = 缺陷八面体配位**（GST 的母体结构），
**109.5° 附近的峰 = 四面体配位**（Ge 的少数态）。两者的相对权重是 GST 非晶态结构的核心问题，
ADF 与下面的四面体序参量一起读。

### 4.4 配位数 CN

在冻结截断下逐帧统计，输出的是**完整分布**而不只是均值。

| metric_id | 单位 | 含义 |
|---|---|---|
| `L3a.cn.<species>.mean` | – | 平均配位数。Ge/Sb/Te 分别对应"8−N 规则"下的 4/3/2 与八面体化的偏离；`all` 是总平均 |
| `L3a.cn.<species>.jensen_shannon` | – | **完整 CN 分布**的距离，不只是均值。均值相同而分布不同 = 欠配位与过配位互相抵消，是被均值掩盖的失效模式 |

### 4.5 同极键（chemical order）

| metric_id | 单位 | 含义 |
|---|---|---|
| `L3a.homopolar.<X>` | 键数占比 | X–X 键占全部键的比例（`Ge`/`Sb`/`Te`/`total`）。**这是化学序的定量指标**：GST225 的理想结构里 Ge/Sb 只与 Te 成键，同极键是"错键"。它对结晶速度、能隙、电阻对比度都有直接影响，是相变存储材料最要紧的结构量之一 |

> 本轮实测的核心结论就落在这里：MACE-MPA-0 的 Ge–Ge 同极键 0.0191 → 0.0319（**+67%**），
> 总同极键 +15%，而六个 RDF 首峰位置**全部**落在统计噪声内——**键长学对了，化学序学错了**。
> 液态 1985 K 独立复现同一签名，说明是可迁移的系统性偏差。

### 4.6 四面体序参量（Chau-Hardwick q）

`q = 1 − (3/8) Σ_{i<j} (cosθ_ij + 1/3)²`，取最近的 4 个成键近邻。
`q = 1` 为理想四面体，`q = 0` 为理想气体值。

| metric_id | 单位 | 含义 |
|---|---|---|
| `L3a.tetrahedral_q.Ge.mean` | – | Ge 的平均四面体度；`extra` 里另带 `fraction_q>0.8`（真正四面体配位的 Ge 占比） |
| `L3a.tetrahedral_q.Ge.wasserstein1` | – | q 分布的整体差异 |

物理：Ge 在 a-GST 中的**八面体↔四面体二态**是 umbrella-flip 结晶机制的核心。
四面体 Ge 占比错了，结晶动力学一定错。

---

## 5. L3b — 中程序与拓扑（`l3b_medium_range.py`，tier A）

**在问什么**：短程都对之后，**几个键以外**的连接方式对不对。
这一级的量对"看起来差不多但结晶行为完全不同"的结构最敏感。

### 5.1 环统计

算法**只实现一种**并冻结：**King 最短路径判据**——对每个原子 i 与它的每对成键近邻 (j,k)，
找避开 i 的最短 j→k 路径，与 i 一起闭合成环；按节点集合去重，截断到 `max_ring_size = 12`。
（换算法会改变每一个环数，因此算法名与最大环尺寸写进 `StructureParams` 并回显到每条指标。）

| metric_id | 单位 | 含义 |
|---|---|---|
| `L3b.rings.size<N>` | rings/atom | 每原子 N 元环数量（N = 3…12）。**四元环是 GST 立方序的代理量**：ABAB 四元环是从非晶到 rocksalt 结晶的成核单元。环分布向大环移位 = 网络更"松散无序"，结晶更慢 |
| `L3b.four_ring_planarity.<reference\|model>` | Å rms 面外偏离 | 四元环的均方面外偏离（0 = 完全共面）。**平面性才是"立方型"四元环的判据**：数量对但严重褶皱的四元环不是结晶前体 |

> 本轮：3/5/6/12 元环偏少，7/10 元环偏多——拓扑向大环移位，与同极键偏高是同一个化学序问题的两面。

### 5.2 ALTBC（角度受限三体关联）

对每个中心原子，取所有夹角落在 180° ± 25° 内的键对，把它们的两个键长
`(d1, d2)` 打进二维直方图。

| metric_id | 单位 | 含义 |
|---|---|---|
| `L3b.altbc.asymmetry` | Å | 近共线键对的平均 `\|d1 − d2\|`。**这是 Peierls 畸变的定量测量**：GST 的共振键合体系里，共线的三原子链自发形成"一长一短"的键对（对角线外权重）。模型若把八面体过度对称化，权重堆到对角线上，asymmetry 偏小——直接对应电子结构与光学对比度的错误 |

### 5.3 结构因子 S(q)

**直接对模拟盒的倒格矢求和**，不做 g(r) 的傅里叶变换：22 Å 的盒子在
q < 2π/r_max 区域的变换结果非物理（本体系实测 S(0.4) = 6.7、S(0.7) = −0.6）。
默认中子权重（相干散射长度 Ge 8.185 / Sb 5.57 / Te 5.80 fm）。

| metric_id | 单位 | 含义 |
|---|---|---|
| `L3b.fsdp.q_fsdp` / `S_fsdp` / `fwhm` | 1/Å, –, 1/Å | 第一尖锐衍射峰（FSDP，搜索窗 0.8–1.7 1/Å）的位置/高度/宽度。FSDP 是**中程序的实验可观测量**（对应 ~2π/q ≈ 6 Å 的准周期性）。当峰落在窗口边缘或窗口下界低于盒子能表达的最小波矢 `2π/L` 时，标 `resolved=False` 并把该指标置 `skipped`——306 原子 GST 胞恰好在这个极限上，本轮三条 FSDP 指标全部 `skipped`，这是**如实报告分辨率限制**而非失败 |
| `L3b.principal_peak.q_pp` / `S_pp` | 1/Å, – | 主峰（~2.2 1/Å，搜索窗 1.7–3.2）的位置与高度。它被这个盒子**充分分辨**，因此是 S(q) 侧真正可用的对比量 |

---

## 6. L3 不确定度层（`uncertainty.py`）

**在问什么**：L3 每个状态点产出上百个数，没有误差棒就无法区分"模型真错了"与
"轨迹太短"。本模块提供两个**可测**的误差尺度，并明确声明**二者都不是 `sigma_ref`**，
因此不参与判级。

| 尺度 | 定义 | 回答的问题 |
|---|---|---|
| **分块 SEM** | 把一条轨迹切成 5 个**连续**块（不是随机抽样——MD 帧强自相关，随机抽样会严重低估误差），在每块上算同一观测量，取块间均值的标准误 | "这个数收敛了吗？"（**同一条链内的统计误差**） |
| **种子散布** | 只改随机种子的两条 MLIP 轨迹上同一观测量的极差 | "同一个模型换个种子会不会偶然产生这个差？"（**固定物理下的链间涨落**，测在模型侧） |

`l3_scalars()` 定义被检验的 33 个标量：`cn_mean.{Ge,Sb,Te,all}`、
`homopolar.{Ge,Sb,Te,total}`、`tetrahedral_q.Ge.{mean, fraction_gt_0.8}`、
`rdf.<pair>.{r_peak, g_peak}`（6 对 ×2）、`rings.size3..12`、`altbc.asymmetry`。

判定（`compare_with_uncertainty`，z 阈值 2.0）：

| verdict | 含义 |
|---|---|
| `resolved` | 差值同时超过合并块 SEM（×z）**与**模型自身的种子散布 → 在现有统计量下是真实差异 |
| `within_statistical_noise` | 差值不超过合并块 SEM → 轨迹太短，根本说不出两边有差别 |
| `within_seed_scatter` | 统计显著，但不超过同一模型换种子就能产生的涨落（该散布还被 SEM 兜底，防止两种子的极差偶然接近 0 制造虚假显著性）→ **不可归因于模型误差** |
| `unavailable` | 有一侧估计缺失或非有限 |

> 本轮 300 K 非晶的 33 个观测量：**18 个可分辨、3 个统计显著但落在种子散布内、12 个仍埋在统计噪声里**。
> 上一节引用的所有结论都只来自那 18 个。

---

## 7. L4 — 动力学与输运（`l4_dynamics.py`，tier A）

**在问什么**：模型跑出来的**原子迁移速率**对不对。扩散系数跨数量级变化，
所以比较在 log 空间做。

### 7.1 守卫（先于任何数字）

| metric_id | 单位 | 含义 |
|---|---|---|
| `L4.equilibrium_segment@<state>.reference` / `.model` | bool | **拒绝把升降温轨迹当平衡扩散参考**。协议里温度写成 `(start, end)` 二元组的直接判失败；有逐帧温度时检查前后半段均温漂移 ≤ 100 K。规范明确现有 GST225 升降温段不能用于平衡扩散，所以这条写在模块里而不是 notebook 的注释里 |
| `L4.unwrap_safety@<state>` | 半个最短晶胞边长 | 相邻存帧间的最大位移。XDATCAR 存的是**折叠**的分数坐标，解缠绕靠最小镜像累加；该值 ≥ 0.5 表示采样步长太粗，解缠绕有歧义，MSD 不可信 |
| `L4.fit_window@<state>` | fs | 拟合窗口的**绝对**时间范围（默认取双方共同 lag 跨度的 20%–80%）。用绝对窗口而非比例窗口，是为了防止较短的模型轨迹被悄悄拟合到不同的时间区间上 |

### 7.2 扩散

MSD 按元素分解、多时间原点平均（每个 lag 的原点数一并返回，便于看统计量在哪里耗尽）。
`D = slope/6`（三维），`1 Å²/fs = 0.1 cm²/s`。

| metric_id | 单位 | 含义 |
|---|---|---|
| `L4.diffusion.<species>.D` | cm²/s | 自扩散系数（Ge/Sb/Te/all）。`extra` 里带参考侧与模型侧的完整拟合信息 |
| `L4.diffusion.<species>.delta_log10D` | decades | `log10(D_model) − log10(D_ref)`。**这是 L4 的计分量**：log 空间对高估与低估对称，0.3 decades = 2 倍 |
| `L4.diffusion_reference.<species>.D` | cm²/s | 无模型轨迹时只报参考值 |

**扩散区判定三档**（写在 `DiffusionFit` 里，随 `failure_reason` 输出）：
用窗口内的 **log–log 斜率**判断是否真的进入扩散区（弹道区斜率→2，笼闭区→0，扩散区→1）：
`|slope−1| ≤ tol` → `diffusive`；`≤ 2·tol` → `marginal`；否则 `not diffusive`（判 `failed`）。
另外报告 **D 在三个子窗口间的相对离散** `d_window_spread`——**这才是"D 已收敛"的证据**。

> 本轮：斜率 1.22（AIMD）/ 1.26（MLIP），落在 `marginal`；MLIP 侧只有 4.8 ps，扩散区勉强达到；
> 子窗口离散 7–9%，即数值本身收敛到约 10%。**结论必须连窗口与斜率一起引用。**

---

## 8. L5 — 过程与路径（`l5_pathway.py`，tier A）

**在问什么**：让模型跑**与 AIMD 完全相同的温度程序**（熔淬），比较的不只是终态，
而是**沿程**的结构演化，并定位模型**第一次**离开参考轨道的位置。

### 8.1 沿程观测量（`PATHWAY_OBSERVABLES`）

在程序进度 0→1 上取 40 个采样点，每点对 8 帧连续帧做平均，窗内散布保留为该点的 σ：

`mean_cn_all`、`mean_cn_{Ge,Sb,Te}`、`four_rings_per_atom`、`tetrahedral_fraction`、
`rdf_first_peak_A`、`volume_per_atom_A3`、`potential_energy_per_atom_eV`。

### 8.2 容差带：用参考自己的涨落，不用魔数

`band = max(sigma_factor × 参考窗内散布, floor)`，默认 `sigma_factor = 3`。
固定容差在这里行不通——例如 Ge-Te RDF 峰位的单帧散布（~0.25 Å）本身就比任何
物理上有意义的位移大好几倍。`PATHWAY_TOLERANCE_FLOORS` 给每个观测量一个下限，
防止某处散布偶然很小把带压成 0。

| metric_id | 单位 | 含义 |
|---|---|---|
| `L5.pathway.<obs>.first_divergence_progress` | 程序进度占比 (0–1) | 模型连续 **2** 个采样点离开 3σ 带的**第一个**位置。要求连续两点，避免单个噪声点被判成偏离。`extra` 带 `max_delta` 与 `max_delta_in_sigma_units`。**用途**：`failure_export` 据此导出偏离点**前后的连续帧**，而不是只导出最终那个异常结构——补数据要补在偏离发生的地方 |
| `L5.pathway.<obs>.final_state_error` | 观测量自身单位 | 终态差值；`extra.within_reference_scatter` 说明它是否仍在参考散布内 |
| `L5.pathway_reference.<obs>.final` | | 无模型轨迹时的参考终值 |

### 8.3 当前的可归因性限制（必须随 L5 结论引用）

用第二条不同种子的 MLIP 淬火做对照，`MLIP vs AIMD` 与 `MLIP(seed0) vs MLIP(seed1)`
的最大偏差之比为 **0.69–1.20**：本级标出的"偏离"**全部落在同一模型换种子就能产生的
涨落之内**。要让 L5 变成可归因的结论，需要的是**独立参考 replicas**（TODO.md 第 8 条），
而不是更长的单条轨迹。

---

## 9. 跨模型排名指标（`ranking.py`）

**这不是判级**。判级需要 `sigma_eff`，仍然不可用。这里做的是**相对比较**：
五个模型对同一参考、同一协议、同一套冻结后处理参数、同一条代码路径，
因此它们的误差彼此**直接可比**，即使谁都拿不到绝对等级。

打分公式：

```
score_i = 100 × (x_best + floor) / (x_i + floor)
```

`floor` 是"差异不再有物理意义"的尺度——有实测统计噪声时用它，否则用写死在
`INDICATORS` 里的显式物理分辨率。没有 floor 的话，一个所有模型都基本精确的指标
会靠数值尘埃主宰排名。方向变体：`lower`（默认，越小越好）、`higher`、
`gap`（安全裕度型，看与最好者的差距而非比值）、`pass`（0–1 通过率）；
`transform="one_minus"` 把 R² 之类的拟合优度转成误差 `1−R²` 再打分。

30 项指标（含权重）：

| 级别 | 指标 | 权重 | floor | 说明 |
|---|---|---|---|---|
| L0 | force MAE | 2.0 | 0.02 eV/Å | 免能量零点的精度陈述 |
| L0 | energy MAE | 1.5 | 5 meV/atom | 扣除每组成常数偏移后 |
| L0 | stress RMSE | 1.0 | 0.1 meV/Å³ | tier C，Virial 约定只经审计 |
| L0 | force 1−R² | 1.0 | 0.01 | 按未解释方差打分 |
| L0 | force MAE worst stage | 0.5 | 0.02 eV/Å | 五个轨迹阶段中最差者 = 迁移性 |
| L1 | L1 gate | 2.0 | – | 二值门控；只说过没过，不说差多少 |
| L1 | PEC curves passing L1 | 1.5 | – | 六条曲线的通过比例，比门控细 |
| L1 | NVE drift | 2.0 | 0.05 meV/atom/ps | tier A |
| L1 | energy-force consistency | 1.0 | 5e-3 | floor 即框架自身的通过阈值 |
| L1 | PEC repulsive walls | 1.5 | – | 六条曲线中排斥核单调的比例 |
| L1 | 2000 K min distance | 1.0 | 0.05 Å | `gap` 型，越大越安全 |
| L1 | lattice scan smoothness | 0.5 | 1.0 | |
| L1 | timestep drift spread | 1.0 | 0.05 | dt 3.0 vs 1.5 fs |
| L2 | relaxed volume error | 2.0 | 0.5 % | |
| L2 | relaxed cell-length error | 1.0 | 0.5 % | |
| L2 | EOS V0 error | 1.5 | 0.5 % | |
| L2 | EOS B0 error | 1.5 | 5 % | 曲率，比 V0 敏感得多 |
| L2 | phase order | 1.5 | – | 只记符号对不对 |
| L3 | partial RDF W1 | 2.0 | 0.02 Å | 六对 × 两个状态点：键长陈述 |
| L3 | ADF JS | 1.5 | 0.02 | |
| L3 | CN mean error | 2.0 | 0.05 | |
| L3 | homopolar bond error | 2.0 | 0.004 | 化学序签名 |
| L3 | tetrahedral q W1 | 1.0 | 0.02 | 仅非晶态 |
| L3 | ring distribution error | 1.5 | 0.02 rings/atom | 中程拓扑 |
| L3 | ALTBC asymmetry error | 1.0 | 0.02 Å | Peierls 畸变 |
| L3 | principal peak error | 0.5 | 0.02 1/Å | |
| L4 | diffusion Δlog10D | 3.0 | 0.05 decades | 全表最高权重 |
| L4 | equilibrium segment | 1.0 | – | 模型 MD 自身得先是平衡段 |
| L5 | pathway endpoint error | 2.0 | 0.02 | |
| L5 | first divergence | 1.0 | 0.05 | `higher`：沿参考路径走得越远越好 |

级别权重：`L0 0.20 / L1 0.15 / L2 0.15 / L3 0.25 / L4 0.15 / L5 0.10`
（L3 最重，因为第一阶段的目标是非晶 GST225 的结构；L1 按门控而非质量陈述加权）。

**L1 门控不截断总分**：规范第 6 条管的是"别把机时花在门后的 L4/L5"，而排名运行时
机时已经花掉了。门控关闭真正意味着**该模型的 L2–L5 数字只能作参考**——
看排名前先看门控列。

---

## 10. metric_id 命名约定

```
L<级别>.<族>.<对象>.<字段>@<状态点>
```

- 级别：`L0` `L1` `L2` `L3a` `L3b` `L4` `L5`
- `@<状态点>`：L3/L4/L5 特有（`amorphous_300K` / `liquid_2000K` / `melt_quench`），
  用于防止不同状态点的数被并到一起
- `<族>_reference.*`：**没有模型轨迹时**只输出参考侧的值（如 `L3a.rdf_reference.Ge-Te.r_peak`），
  与模型-参考对比量在 id 层面就分开
- L0 的分箱指标用 `<族>.<标签>=<组>` 形式（`L0.force_rmse.stage=1-2000K`）

---

## 11. 已知限制（引用任何数字前先读）

1. **所有模型-参考对比都标 `provisional`**：每个状态点/相只有一条参考，`sigma_ref` 未知；
   L0 还缺 `d_protocol`（需要 `results/bridge_set/` 那 24 个 SCF）。框架里没有正式 grade。
2. **L0 的 stress 永不计分**：参考 Virial 协议不可溯源，只做约定审计。
3. **L3b 的 FSDP 在 306 原子胞上不可分辨**，三条指标 `skipped`；用主峰代替。
4. **L3 液态的键截断需要显式选择**（见 4.0），`cutoff_source_state` 必须随结论引用。
5. **L4 只有一个温度点**，无 Arrhenius；拟合窗口斜率落在 `marginal` 档。
6. **L5 的"首次偏离"目前不可归因**（见 8.3）。
7. **立方相 DFT EOS 仍在排队**，`L2.eos.cubic.*` 目前是 tier S（仅模型自身量），
   到位后 `run_l2.py` 自动读入，无需改代码。
8. 306 原子参考的数密度 0.02852 at/Å³ 比项目标准 0.0309 at/Å³ 低 8.2%，
   与 phase-1 melt–quench 链跨用必须声明。
