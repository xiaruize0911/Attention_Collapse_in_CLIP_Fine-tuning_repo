# Controlled CLIP Attention-Heatmap Drift Report

## Scope

This report summarizes the completed matched-learning-rate controlled study for CLIP ViT-B/32 using only the validated outputs in `outputs/controlled_heatmap_drift/`.

- Controlled matrix size: `80` runs
- Datasets: `EuroSAT`, `Oxford-IIIT Pets`
- Methods: `Full FT`, `LoRA r=8`
- Shared LR grid: `1e-6`, `5e-6`, `1e-5`, `5e-5`
- Seeds: `7`, `11`, `19`, `42`, `123`
- Primary transfer benchmark: `CIFAR-100` zero-shot

Primary source files:

- `outputs/controlled_heatmap_drift/run_summaries.csv`
- `outputs/controlled_heatmap_drift/run_summaries.json`
- `outputs/controlled_heatmap_drift/analysis_summary.json`

## Core Claim

The completed controlled study supports a narrower and more defensible claim than "attention collapse":

- attention heatmap drift depends strongly on learning rate
- LoRA is usually more structurally conservative than Full FT at matched learning rates
- better retention of pretrained attention support is associated with better CIFAR-100 zero-shot retention
- method-only averages can be misleading on Pets because low-LR LoRA underfits badly

## Figures

- Matched-LR summary panels: `outputs/controlled_heatmap_drift/figures/controlled_lr_panels.png`
- Per-layer entropy heatmaps: `outputs/controlled_heatmap_drift/figures/controlled_per_layer_entropy_heatmaps.png`

## Zero-Shot Baseline

- Pretrained CLIP CIFAR-100 zero-shot: `60.22%`

## Top Findings

1. EuroSAT shows the clearest method split.
   Full FT moves from mild broadening at `1e-6` (`+1.83%` entropy) to strong contraction at `5e-5` (`-3.99%`), while LoRA remains entropy-positive across the full matched grid (`+0.68%` to `+1.50%`).

2. LoRA preserves CIFAR-100 transfer much better than Full FT at matched LRs.
   Averaged across the EuroSAT grid, Full FT reaches `11.28%` CIFAR-100 zero-shot while LoRA reaches `45.13%`.
   Averaged across the Pets grid, Full FT reaches `8.54%` while LoRA reaches `58.01%`.

3. Pets is more nuanced than a simple method ranking.
   Full FT causes stronger structural contraction on Pets on average (`-2.32%` entropy, `-5.18%` ERF) than LoRA (`-0.33%` entropy, `-1.44%` ERF), but low-LR LoRA dramatically underfits the in-domain task.
   The best Pets LoRA configuration still edges out the best Pets Full FT configuration in validation accuracy (`92.18%` vs `91.66%`), but only at the high end of the shared LR grid.

4. Run-level entropy drift tracks transfer retention much more strongly than in-domain accuracy.
   Dataset-level entropy-vs-CIFAR correlation is `0.61` on EuroSAT and `0.90` on Pets, while entropy-vs-accuracy correlation is weak to moderate (`-0.14`, `-0.43`).

5. The strongest contraction appears in later layers, especially under high-LR Full FT.
   On Pets at `5e-5`, Full FT reaches a mean layer-12 entropy shift of `-20.29%`, compared with `-11.53%` for LoRA at the same LR.

## Matched-LR Summary: EuroSAT

| Method | LR | Mean Best Val Acc | Mean Delta Entropy | Mean CIFAR-100 |
| --- | --- | ---: | ---: | ---: |
| Full FT | `1e-6` | `98.82%` | `+1.83%` | `15.90%` |
| Full FT | `5e-6` | `99.11%` | `+0.41%` | `17.52%` |
| Full FT | `1e-5` | `99.13%` | `-0.12%` | `10.01%` |
| Full FT | `5e-5` | `98.79%` | `-3.99%` | `1.68%` |
| LoRA r=8 | `1e-6` | `92.07%` | `+0.68%` | `56.26%` |
| LoRA r=8 | `5e-6` | `97.35%` | `+1.38%` | `44.64%` |
| LoRA r=8 | `1e-5` | `98.13%` | `+1.50%` | `41.69%` |
| LoRA r=8 | `5e-5` | `98.83%` | `+1.18%` | `37.93%` |

Interpretation:

- Full FT reaches the highest EuroSAT validation accuracy near `1e-5`, but it does so with much lower transfer retention than LoRA.
- LoRA pays an in-domain accuracy cost at the smallest LR, then closes most of the gap by `5e-5` while still preserving much more zero-shot capability.

## Matched-LR Summary: Oxford-IIIT Pets

| Method | LR | Mean Best Val Acc | Mean Delta Entropy | Mean CIFAR-100 |
| --- | --- | ---: | ---: | ---: |
| Full FT | `1e-6` | `88.40%` | `-1.57%` | `18.96%` |
| Full FT | `5e-6` | `90.40%` | `-1.86%` | `6.38%` |
| Full FT | `1e-5` | `91.20%` | `-2.21%` | `5.01%` |
| Full FT | `5e-5` | `85.75%` | `-3.63%` | `3.83%` |
| LoRA r=8 | `1e-6` | `19.13%` | `+0.01%` | `60.24%` |
| LoRA r=8 | `5e-6` | `82.08%` | `-0.20%` | `60.57%` |
| LoRA r=8 | `1e-5` | `89.93%` | `-0.38%` | `59.34%` |
| LoRA r=8 | `5e-5` | `91.91%` | `-0.75%` | `51.91%` |

Interpretation:

- On Pets, LoRA only becomes competitive once LR is large enough to avoid underfitting.
- Once LR is high enough, LoRA remains structurally milder than Full FT and preserves far more CIFAR-100 transfer.

## Aggregate Method Means

| Dataset | Method | Mean Delta Entropy | Mean Delta ERF | Mean Best Val Acc | Mean CIFAR-100 |
| --- | --- | ---: | ---: | ---: | ---: |
| EuroSAT | Full FT | `-0.47%` | `-1.97%` | `98.96%` | `11.28%` |
| EuroSAT | LoRA r=8 | `+1.18%` | `+1.55%` | `96.59%` | `45.13%` |
| Pets | Full FT | `-2.32%` | `-5.18%` | `88.94%` | `8.54%` |
| Pets | LoRA r=8 | `-0.33%` | `-1.44%` | `70.76%` | `58.01%` |

Interpretation:

- These averages are useful as a structural summary.
- They should not be presented without the LR tables because they hide the strong underfitting regime for low-LR LoRA on Pets.

## Recommended Paper Framing

- Use "attention structural drift" or "attention heatmap drift" instead of "attention collapse" as the headline claim.
- Present method, dataset, and learning rate jointly.
- Treat CIFAR-100 as the main transfer benchmark and keep its values adapter-aware.
- Avoid claims that LoRA perfectly preserves zero-shot transfer.
- Frame attention metrics as descriptive structural measurements, not causal explanations.

## Recommended Main Narrative

1. Matched learning rates reveal that optimization scale is a first-order control variable for CLIP heatmap drift.
2. LoRA is more conservative than Full FT on structure and zero-shot transfer across the completed controlled grid.
3. The method comparison is strongest when reported as a method-by-LR interaction, especially on Pets.
4. Later-layer contraction under high-LR Full FT is the most visible structural failure mode in the completed matrix.

## Remaining Cleanup Targets

- Update the longer paper sources to cite this controlled report and these controlled-study figures.
- Remove or quarantine older manuscript claims that rely on the pre-controlled 21-run matrix, Flowers102, or non-matched LR comparisons.
