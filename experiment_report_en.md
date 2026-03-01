# Attention Collapse in CLIP Fine-tuning: LoRA vs Full Adaptation (English Report)

## Abstract

This report studies how CLIP ViT-B/32 attention structure changes during downstream fine-tuning. We compare full fine-tuning (Full FT), LoRA, and regularized Full FT across 21 experiment configurations. We track four structural metrics (entropy, ERF@0.95, Gini, head diversity), test significance (Welch’s t-test, Mann–Whitney, bootstrap CIs, ANOVA), and link structural changes to zero-shot transfer.

Main findings:
1. Full FT tends to reduce attention entropy, while LoRA tends to increase it on EuroSAT.
2. The Full FT vs LoRA per-layer entropy difference is significant (Welch’s $t=-2.50$, $p=0.019$, Cohen’s $d=0.71$).
3. Higher LR causes stronger collapse (Pearson $r=-0.893$, $p=0.041$; Spearman $\rho=-1.0$).
4. Entropy Floor ($\lambda=0.1$) is the only individually significant regularizer ($p=0.023$).
5. Collapse is not significantly concentrated in early/middle/late layer groups (ANOVA $p>0.46$).

---

## 1. Motivation and Questions

CLIP is pre-trained for broad image–text alignment, but downstream adaptation can distort internal attention geometry.
We test:
- **RQ1:** Does fine-tuning reduce attention entropy systematically?
- **RQ2:** Does LoRA preserve structure better than Full FT?
- **RQ3:** How does collapse relate to zero-shot transfer degradation?
- **RQ4:** Can regularizers mitigate collapse without hurting task accuracy?
- **RQ5:** Which control knobs matter most (LR, rank, frozen layers)?

---

## 2. Metrics: Definition, Intuition, and Why They Matter

We evaluate [CLS]→patch attention vectors (49 patches) across 12 layers.

### 2.1 Attention Entropy

$$
H(a) = -\sum_{j=1}^{49} a_j \ln a_j
$$

- **Meaning:** concentration vs spread of attention over patches.
- **Higher $H$:** more distributed attention.
- **Lower $H$:** more peaky attention (collapse tendency).

### 2.2 ERF@0.95 (Effective Receptive Field)

Sort attention weights descending and find smallest fraction of tokens needed to accumulate 95% mass:

$$
\mathrm{ERF}_{0.95} = \frac{k_{0.95}}{49}
$$

- **Higher ERF:** broader spatial support.
- **Lower ERF:** model relies on fewer patches.

### 2.3 Gini Coefficient of Attention

$$
G = \frac{2\sum_{i=1}^{N} i\,x_{(i)}}{N\sum_{i=1}^{N} x_{(i)}} - \frac{N+1}{N}, \quad N=49
$$

- $x_{(i)}$ is sorted attention weight.
- **Higher $G$:** more unequal (more concentrated) attention.

### 2.4 Head Diversity

$$
D = 1 - \frac{1}{\binom{H}{2}}\sum_{i<j}\cos(h_i,h_j), \quad H=12
$$

- Compares attention maps across heads.
- **Higher $D$:** heads specialize differently.
- Captures a different axis than entropy/ERF/Gini.

### 2.5 Statistical Quality Controls

We report:
- Welch’s t-test (unequal variance)
- Mann–Whitney U (non-parametric)
- Bootstrap 95% CIs (10,000 resamples)
- Cohen’s $d$ effect size
- One-way ANOVA for layer position groups
- Pearson + Spearman correlations

This combination guards against overclaiming from a single test family.

---

## 3. Experimental Setup

- **Model:** `openai/clip-vit-base-patch32`
- **Hardware:** NVIDIA A40 (46GB)
- **Framework:** PyTorch + Transformers + PEFT
- **Datasets:** EuroSAT, Oxford-IIIT Pets, CIFAR-100 (zero-shot)
- **Training:** AdamW, cosine schedule, warmup, FP16, fixed seed 42
- **Scale:** 21 experiment configurations

---

## 4. Results with Figures in Place

### 4.1 Baseline Structure (Pretrained CLIP)

Pretrained CLIP already shows layer-wise heterogeneity: earlier layers are more diffuse; deeper layers are more concentrated.

![Fig 2. Baseline structure](outputs/figures/fig2_baseline_structure.png)

### 4.2 Full FT vs LoRA Structural Trajectory

- Full FT (EuroSAT) mild entropy drop; Full FT (Pets) stronger drop.
- LoRA on EuroSAT gives positive entropy deltas.
- Statistical separation is significant: $p=0.019$, $d=0.71$.

![Fig 5. Full FT vs LoRA (EuroSAT)](outputs/figures/fig5_fullft_vs_lora.png)

![Fig 5b. Full FT vs LoRA (Pets)](outputs/figures/fig5b_fullft_vs_lora_pets.png)

![Fig 11. Statistical summary](outputs/figures/fig11_statistical_summary.png)

### 4.3 Learning Rate is the Main Collapse Driver

Entropy collapse scales strongly with learning rate:
- $10^{-6}$: entropy increase
- $10^{-5}$: near-neutral
- $5\times10^{-5}$, $10^{-4}$: strong collapse

![Fig 6. LR sweep](outputs/figures/fig6_lr_sweep.png)

Interpretation: larger update magnitudes likely push attention logits into higher-contrast regimes faster, reducing entropy and ERF while increasing Gini.

### 4.4 Zero-shot Transfer and Structural Shift

Under current evaluation outputs, Full FT dramatically lowers CIFAR-100 zero-shot accuracy, while LoRA entries stay near baseline.

![Fig 7. Collapse vs zero-shot](outputs/figures/fig7_collapse_vs_zeroshot.png)

![Fig 15. Extended zero-shot](outputs/figures/fig15_zeroshot_extended.png)

### 4.5 Per-layer, Dynamics, and Correlations

Per-layer heatmap and dynamics show collapse can emerge early at high LR and is not confined to one depth band.

![Fig 10. Per-layer entropy change](outputs/figures/fig10_per_layer_delta_heatmap.png)

![Fig 12. Training dynamics](outputs/figures/fig12_training_dynamics.png)

Cross-metric correlation confirms entropy/ERF/Gini are tightly coupled, while head diversity is relatively orthogonal.

![Fig 13. Metric correlations](outputs/figures/fig13_metric_correlations.png)

### 4.6 Convergence and Overall Dashboard

All 21 runs satisfy the late-epoch stability criterion used in this project.

![Fig 14. Convergence check](outputs/figures/fig14_convergence.png)

![Fig 16. Dashboard](outputs/figures/fig16_dashboard.png)

![Fig 17. Gini evolution](outputs/figures/fig17_gini_evolution.png)

---

## 5. Clearer Interpretation of Why We See These Effects

### 5.1 Why Full FT can collapse attention

Full FT updates all vision backbone weights, so optimization pressure from task loss can directly reshape attention projections. At high LR, this reshaping is abrupt, yielding:
- lower entropy,
- narrower ERF,
- higher Gini.

### 5.2 Why LoRA often looks structurally safer

LoRA constrains updates to low-rank adapters, limiting representational drift of backbone weights. This can reduce over-concentration pressure and preserve broader attention patterns.

### 5.3 Why accuracy may stay high even when structure shifts

In-domain classification can improve with specialized features, while out-of-domain transfer may degrade. Hence weak entropy–accuracy correlation but strong collapse–transfer degradation patterns.

---

## 6. Code and Reasoning Audit (Double-check Section)

### 6.1 What was verified

- Reported statistics match `outputs/metrics/statistical_tests.json`.
- Extended zero-shot values match `outputs/metrics/extended_zs_*.json`.
- Figure files exist and correspond to referenced analyses.

### 6.2 Important methodological caveat discovered

In `run_all_experiments.py` and `enhanced_analysis.py`, zero-shot loading logic extracts only keys with `vision_model.` prefix into a plain `CLIPModel` vision backbone. For LoRA checkpoints, adapter (`lora_*`) contributions are not explicitly merged in this path.

**Implication:** the “LoRA perfectly preserves zero-shot” statement should be interpreted as valid for the **current evaluation protocol**, but may overstate adapter-active zero-shot retention. A dedicated adapter-active evaluation path is recommended before making stronger causal claims.

### 6.3 Recommended follow-up validation

1. Evaluate LoRA checkpoints with adapters active end-to-end for image embedding extraction.
2. Compare base-only vs adapter-active zero-shot on CIFAR-100 and a second robust benchmark.
3. Report both numbers side-by-side.

---

## 7. Conclusions (Conservative, Evidence-based)

1. Full FT and LoRA show statistically different structural behavior ($p=0.019$, $d=0.71$).
2. LR is the most sensitive collapse control in tested settings.
3. Entropy-floor regularization ($\lambda=0.1$) is statistically significant while keeping high task accuracy.
4. Layer-position ANOVA does not support a strong early/mid/late localization claim.
5. Entropy/ERF/Gini are coherent collapse indicators; head diversity captures a different factor.
6. Zero-shot conclusions for LoRA should explicitly mention the current loader/protocol caveat.

---

## 8. Citations

- Radford et al., 2021 (CLIP)
- Dosovitskiy et al., 2021 (ViT)
- Hu et al., 2022 (LoRA)
- Abnar & Zuidema, 2020 (attention rollout)
- Voita et al., 2019 (head behavior)
- Wortsman et al., 2022 (robust fine-tuning)
- Zhai et al., 2023 (entropy collapse)
- Kirkpatrick et al., 2017; McCloskey & Cohen, 1989 (forgetting)

---

## 9. Reproducibility Pointers

- Code: `run_all_experiments.py`, `enhanced_analysis.py`, `analyze_and_visualize.py`
- Metrics output: `outputs/metrics/`
- Figures: `outputs/figures/`
- Checkpoints: `outputs/checkpoints/`
