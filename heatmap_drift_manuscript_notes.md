# Manuscript Reframe Notes: Controlled Attention-Heatmap Drift

## Purpose

This note records the paper-facing rewrite required after the repo moved to the
controlled heatmap-drift design. It avoids rewriting the current manuscript with
invented numbers before the new runs are complete.

## Title Direction

Prefer titles built around:

- `attention heatmap drift`
- `structural drift`
- `controlled adaptation`
- `matched learning-rate comparison`

Avoid titles that imply:

- first-ever heatmap analysis
- universal attention collapse
- method-only causality without optimization control

## Introduction Changes

- Acknowledge prior CLIP / VLM attention-visualization and explanation work.
- State that the gap is not the existence of heatmaps, but the lack of a clean method-versus-LR comparison during adaptation.
- Position the contribution as a controlled empirical study of heatmap drift and transfer retention.

## Methods Changes

- Define the main study on `ViT-B/32`, `EuroSAT`, `Oxford-IIIT Pets`, `CIFAR-100`.
- Report the shared LR grid for both Full FT and LoRA.
- State that the evaluation subset for heatmap summaries is fixed across methods and seeds.
- State that zero-shot evaluation for LoRA is adapter-aware.

## Results Changes

- Main quantitative section: run-level summaries by `dataset`, `method`, `lr`.
- Main figures: same-image heatmaps, per-layer drift grids, fixed-subset average heatmaps, drift-versus-transfer summary.
- Keep Gini, head diversity, and rollout as supporting triangulation, not as the lead story.

## Discussion Changes

- If LR dominates, say so directly.
- If method differences remain after LR matching, frame them as conditional on optimization scale.
- Do not claim that heatmap drift alone explains downstream behavior; treat it as a descriptive structural signal.
