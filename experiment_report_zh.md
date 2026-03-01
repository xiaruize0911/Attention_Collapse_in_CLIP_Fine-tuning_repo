# CLIP 微调中的注意力坍缩：LoRA 与全量微调对比（中文版）

## 摘要

本报告研究 CLIP ViT-B/32 在下游微调时注意力结构如何变化。我们系统比较了全量微调（Full FT）、LoRA 以及带正则项的 Full FT，共 21 组配置。通过 4 个结构指标（熵、ERF@0.95、Gini、Head Diversity）和统计检验（Welch t、Mann–Whitney、Bootstrap 置信区间、ANOVA、相关分析），我们发现：

1. Full FT 更倾向于降低注意力熵（高学习率下明显坍缩）。
2. LoRA 在 EuroSAT 上整体更倾向于提高熵；与 Full FT 的分层熵变化差异显著（$t=-2.50$, $p=0.019$, $d=0.71$）。
3. 学习率是最强驱动因素（Pearson $r=-0.893$, $p=0.041$；Spearman $\rho=-1.0$）。
4. 熵下限正则（$\lambda=0.1$）是单项显著的结构保持方法（$p=0.023$）。
5. 层位置分组（早/中/晚）ANOVA 不显著（$p>0.46$），说明坍缩并非只发生在某个固定深度段。

---

## 1. 研究动机与问题

CLIP 预训练获得强泛化能力，但下游微调可能改变其内部注意力几何结构，导致泛化能力下降。核心问题：

- **RQ1**：微调是否系统性降低注意力熵？
- **RQ2**：LoRA 是否比 Full FT 更能保留结构？
- **RQ3**：结构坍缩与零样本迁移下降有何关系？
- **RQ4**：正则化能否在保持精度的同时缓解坍缩？
- **RQ5**：学习率、LoRA rank、冻结层数谁最关键？

---

## 2. 指标解释（含数学形式）

我们使用 [CLS]→49 个 patch 的注意力分布，跨 12 层统计。

### 2.1 注意力熵（Attention Entropy）

$$
H(a) = -\sum_{j=1}^{49} a_j \ln a_j
$$

- **高熵**：注意力分布更均匀。
- **低熵**：注意力更尖锐，坍缩倾向更强。

### 2.2 ERF@0.95（有效感受野）

将注意力从大到小排序，累计达到 95% 质量所需 token 比例：

$$
\mathrm{ERF}_{0.95} = \frac{k_{0.95}}{49}
$$

- **更高**：覆盖更广。
- **更低**：依赖更少 patch。

### 2.3 Gini 系数（注意力不均衡度）

$$
G = \frac{2\sum_{i=1}^{N} i\,x_{(i)}}{N\sum_{i=1}^{N} x_{(i)}} - \frac{N+1}{N}, \quad N=49
$$

- **更高**：分布更不均，注意力更集中。

### 2.4 Head Diversity（头间多样性）

$$
D = 1 - \frac{1}{\binom{H}{2}}\sum_{i<j}\cos(h_i,h_j), \quad H=12
$$

- **更高**：不同注意力头更“分工明确”。
- 与熵/ERF/Gini 描述的是不同维度。

### 2.5 统计检验体系

- Welch t-test（方差不齐）
- Mann–Whitney U（非参数）
- Bootstrap 95% CI（1 万次重采样）
- Cohen’s $d$（效应量）
- ANOVA（层位置组间）
- Pearson + Spearman（线性与单调相关）

这能降低“只靠一个检验得结论”的风险。

---

## 3. 实验设置

- **模型**：`openai/clip-vit-base-patch32`
- **硬件**：NVIDIA A40 46GB
- **数据集**：EuroSAT、Oxford-IIIT Pets、CIFAR-100（零样本）
- **训练**：AdamW、余弦退火、warmup、FP16、随机种子 42
- **实验规模**：21 组配置

---

## 4. 结果（图文对应放置）

### 4.1 预训练基线结构

预训练 CLIP 已有明显层间结构差异：浅层更分散，深层更集中。

![图2：基线结构](outputs/figures/fig2_baseline_structure.png)

### 4.2 Full FT vs LoRA 的结构轨迹

- Full FT 在 EuroSAT 为轻微熵下降，在 Pets 上下降更明显。
- LoRA（EuroSAT）总体为熵上升。
- 分层熵变化差异具统计显著性：$p=0.019$，$d=0.71$。

![图5：EuroSAT 上 Full FT vs LoRA](outputs/figures/fig5_fullft_vs_lora.png)

![图5b：Pets 上 Full FT vs LoRA](outputs/figures/fig5b_fullft_vs_lora_pets.png)

![图11：统计汇总](outputs/figures/fig11_statistical_summary.png)

### 4.3 学习率是坍缩主驱动

学习率越高，坍缩越强：
- $10^{-6}$：熵上升
- $10^{-5}$：接近中性
- $5\times10^{-5}, 10^{-4}$：明显坍缩

![图6：学习率扫描](outputs/figures/fig6_lr_sweep.png)

机制解释：较大步长使注意力 logit 更快进入高对比（尖峰化）区域，导致熵降、ERF 收缩、Gini 升高。

### 4.4 零样本迁移与结构变化

当前输出结果显示：Full FT 的 CIFAR-100 零样本显著下降，LoRA 项目接近基线。

![图7：坍缩 vs 零样本](outputs/figures/fig7_collapse_vs_zeroshot.png)

![图15：扩展零样本](outputs/figures/fig15_zeroshot_extended.png)

### 4.5 分层、动态与相关性

高学习率下坍缩早期就出现；分层热图显示并非单层独占。

![图10：分层熵变化热图](outputs/figures/fig10_per_layer_delta_heatmap.png)

![图12：训练动态](outputs/figures/fig12_training_dynamics.png)

相关性结果：熵/ERF/Gini 高相关，Head Diversity 相对独立。

![图13：指标相关矩阵](outputs/figures/fig13_metric_correlations.png)

### 4.6 收敛性与全局总览

全部 21 个实验在本项目收敛判据下通过。

![图14：收敛检查](outputs/figures/fig14_convergence.png)

![图16：综合看板](outputs/figures/fig16_dashboard.png)

![图17：Gini 演化](outputs/figures/fig17_gini_evolution.png)

---

## 5. 结论解释（更清晰版）

### 5.1 为什么 Full FT 更容易坍缩

Full FT 会更新视觉骨干全部参数。任务损失梯度可直接重塑注意力投影矩阵；高学习率下重塑更剧烈，容易产生尖峰化注意力分布。

### 5.2 为什么 LoRA 常表现为“更稳”

LoRA 将更新限制在低秩子空间，降低主干表示漂移幅度，因而更不容易发生强烈集中化。

### 5.3 为什么任务精度高但泛化变差

在域内分类任务中，结构特化可能有利于精度；但域外迁移（零样本）更依赖预训练几何结构，一旦结构改变，迁移能力可能明显下降。

---

## 6. 代码与推理复核（双重检查）

### 6.1 已复核内容

- 报告关键统计与 `outputs/metrics/statistical_tests.json` 一致。
- 扩展零样本结果与 `outputs/metrics/extended_zs_*.json` 一致。
- 图文件存在且与分析流程匹配。

### 6.2 发现的关键方法学注意点

在 `run_all_experiments.py` 与 `enhanced_analysis.py` 中，零样本加载逻辑主要将 `vision_model.` 前缀参数载入到普通 `CLIPModel` 视觉分支；LoRA 的 `lora_*` 适配器项并未在该路径中显式 merge。

**影响**：
“LoRA 完全保留零样本能力”的结论在**当前评估协议**下成立，但对“LoRA 适配器实际启用后的零样本能力”仍需独立验证，结论应保持谨慎。

### 6.3 建议补充验证

1. 使用“适配器启用”的完整路径评估 LoRA 零样本。
2. 同时报告 base-only 与 adapter-active 两组零样本结果。
3. 增加第二个稳定零样本基准数据集做交叉验证。

---

## 7. 保守且可信的结论

1. Full FT 与 LoRA 的结构变化方向显著不同（$p=0.019$, $d=0.71$）。
2. 学习率是最敏感的坍缩控制旋钮。
3. 熵下限正则（$\lambda=0.1$）在个体比较中达到统计显著且精度保持高位。
4. 层位置组间差异不显著，不支持“只在某一段层坍缩”的强断言。
5. 熵/ERF/Gini 构成一致的坍缩观测三联；Head Diversity 描述正交维度。
6. LoRA 零样本结论需明确当前加载协议限制。

---

## 8. 参考文献（核心）

- Radford et al., 2021（CLIP）
- Dosovitskiy et al., 2021（ViT）
- Hu et al., 2022（LoRA）
- Abnar & Zuidema, 2020（Attention Rollout）
- Voita et al., 2019（多头注意力分析）
- Wortsman et al., 2022（鲁棒微调）
- Zhai et al., 2023（熵坍缩）
- Kirkpatrick et al., 2017；McCloskey & Cohen, 1989（灾难性遗忘）

---

## 9. 复现实验路径

- 主流程：`run_all_experiments.py`
- 增强分析：`enhanced_analysis.py`
- 原分析绘图：`analyze_and_visualize.py`
- 指标输出：`outputs/metrics/`
- 图像输出：`outputs/figures/`
- 模型检查点：`outputs/checkpoints/`
