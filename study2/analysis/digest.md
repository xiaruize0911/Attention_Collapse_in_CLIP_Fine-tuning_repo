# Study B digest (66 runs on the primary backbone)

Plus 18 runs on `openai/clip-vit-base-patch16`, analysed separately in the backbone section below.

## Cells (mean over seeds)

| dataset | method | lr | n | test acc | dH % | dH L12 % | C100 ret % | C10 ret % | w-drift | emb-drift | CKA | corrupt acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eurosat | Full FT | 3e-06 | 3 | 96.81±0.07 | +1.53±0.25 | +3.39 | 55.0±4.3 | 66.6 | 0.0022 | 0.632 | 0.844 | 51.6 |
| eurosat | Full FT | 1e-05 | 3 | 97.19±0.33 | +0.99±0.17 | +1.24 | 41.2±2.3 | 57.5 | 0.0042 | 0.726 | 0.802 | 53.5 |
| eurosat | Full FT | 3e-05 | 3 | 95.98±1.36 | +0.06±0.21 | -2.39 | 22.3±4.5 | 45.9 | 0.0083 | 0.730 | 0.762 | 55.5 |
| eurosat | Full FT | 1e-04 | 3 | 94.02±0.43 | -2.34±0.40 | -7.75 | 3.7±0.6 | 15.7 | 0.0222 | 0.731 | 0.731 | 58.8 |
| eurosat | Last block | 1e-05 | 3 | 88.57±0.57 | -0.05±0.01 | -0.67 | 97.1±0.3 | 96.4 | 0.0040 | 0.513 | 0.981 | 41.5 |
| eurosat | Last block | 1e-04 | 3 | 94.38±0.48 | -0.39±0.04 | -4.86 | 83.8±2.2 | 93.2 | 0.0109 | 0.718 | 0.952 | 47.4 |
| eurosat | Linear probe | 1e-03 | 3 | 85.11±0.23 | +0.00±0.00 | +0.00 | 100.0±0.0 | 100.0 | 0.0000 | 0.000 | 1.000 | 37.9 |
| eurosat | LoRA r=8 | 3e-06 | 3 | 45.54±0.80 | +0.01±0.02 | +0.00 | 99.6±0.4 | 99.9 | 0.0003 | 0.007 | 0.999 | 23.7 |
| eurosat | LoRA r=8 | 1e-05 | 3 | 78.83±1.54 | +0.16±0.07 | +0.37 | 98.2±0.7 | 99.7 | 0.0010 | 0.074 | 0.970 | 35.3 |
| eurosat | LoRA r=8 | 3e-05 | 3 | 91.73±0.19 | +0.71±0.08 | +2.04 | 94.5±0.4 | 97.8 | 0.0024 | 0.325 | 0.927 | 47.7 |
| eurosat | LoRA r=8 | 1e-04 | 3 | 95.16±0.44 | +0.98±0.03 | +1.71 | 86.4±1.1 | 92.7 | 0.0042 | 0.530 | 0.900 | 54.9 |
| eurosat | LoRA (frozen proj.) | 1e-05 | 3 | 22.63±5.69 | +0.05±0.10 | +0.25 | 98.6±0.4 | 99.6 | 0.0010 | 0.033 | 0.983 | 14.1 |
| eurosat | LoRA (frozen proj.) | 1e-04 | 3 | 92.67±0.17 | +1.39±0.30 | +3.23 | 76.6±2.6 | 84.6 | 0.0063 | 0.458 | 0.885 | 49.3 |
| pets | Full FT | 3e-06 | 3 | 75.82±2.47 | -0.62±0.26 | -4.95 | 73.5±5.7 | 83.2 | 0.0020 | 0.486 | 0.848 | -- |
| pets | Full FT | 1e-05 | 3 | 87.72±0.20 | -1.23±0.26 | -6.97 | 44.2±8.9 | 59.8 | 0.0046 | 0.672 | 0.785 | -- |
| pets | Full FT | 3e-05 | 3 | 88.26±0.26 | -1.90±0.22 | -10.30 | 21.8±4.0 | 37.6 | 0.0100 | 0.725 | 0.676 | -- |
| pets | Full FT | 1e-04 | 3 | 79.10±0.49 | -3.23±0.20 | -18.51 | 9.2±2.0 | 22.3 | 0.0251 | 0.749 | 0.561 | -- |
| pets | Linear probe | 1e-03 | 3 | 80.50±1.25 | +0.00±0.00 | +0.00 | 100.0±0.0 | 100.0 | 0.0000 | 0.000 | 1.000 | -- |
| pets | LoRA r=8 | 3e-06 | 3 | 6.23±0.93 | -0.00±0.00 | -0.01 | 100.0±0.3 | 100.0 | 0.0001 | 0.002 | 1.000 | -- |
| pets | LoRA r=8 | 1e-05 | 3 | 26.56±1.50 | -0.00±0.01 | -0.08 | 100.1±0.6 | 100.3 | 0.0005 | 0.017 | 1.000 | -- |
| pets | LoRA r=8 | 3e-05 | 3 | 71.75±1.42 | -0.11±0.06 | -2.03 | 100.8±1.8 | 100.4 | 0.0021 | 0.213 | 0.984 | -- |
| pets | LoRA r=8 | 1e-04 | 3 | 89.88±0.47 | -0.35±0.08 | -5.94 | 97.7±1.8 | 96.4 | 0.0049 | 0.591 | 0.935 | -- |

## Paired Full FT vs LoRA (identical dataset, lr, seed)

| dataset | metric | n | Full FT | LoRA | diff | dz | p (t) | p Holm | p (Wilcoxon) |
|---|---|---|---|---|---|---|---|---|---|
| eurosat | delta_entropy_pct | 12 | 0.061 | 0.467 | -0.406 | -0.21 | 0.49 | 0.49 | 0.91 |
| eurosat | cifar100_retention | 12 | 30.527 | 94.697 | -64.169 | -4.14 | 1.8e-08 | 2e-07 | 0.00049 |
| eurosat | cifar10_retention | 12 | 46.391 | 97.550 | -51.158 | -2.86 | 8e-07 | 7.2e-06 | 0.00049 |
| eurosat | test_acc | 12 | 96.000 | 77.815 | +18.185 | +0.85 | 0.013 | 0.026 | 0.0068 |
| eurosat | weight_drift_rel | 12 | 0.009 | 0.002 | +0.007 | +1.08 | 0.0032 | 0.013 | 0.00049 |
| eurosat | cka_mean | 12 | 0.785 | 0.949 | -0.164 | -11.56 | 2.9e-13 | 3.4e-12 | 0.00049 |
| pets | delta_entropy_pct | 12 | -1.747 | -0.115 | -1.632 | -1.84 | 5.4e-05 | 0.00032 | 0.00049 |
| pets | cifar100_retention | 12 | 37.175 | 99.655 | -62.480 | -2.46 | 3.6e-06 | 2.9e-05 | 0.00049 |
| pets | cifar10_retention | 12 | 50.725 | 99.290 | -48.565 | -2.11 | 1.5e-05 | 0.00011 | 0.00049 |
| pets | test_acc | 12 | 82.722 | 48.605 | +34.117 | +0.99 | 0.0055 | 0.017 | 0.0068 |
| pets | weight_drift_rel | 12 | 0.010 | 0.002 | +0.009 | +1.16 | 0.0021 | 0.01 | 0.00049 |
| pets | cka_mean | 12 | 0.718 | 0.980 | -0.262 | -2.90 | 7.2e-07 | 7.2e-06 | 0.00049 |

## Dose-response (Spearman vs log10 lr)

| dataset | method | metric | n | rho | p |
|---|---|---|---|---|---|
| eurosat | Full FT | delta_entropy_pct | 12 | -0.972 | 1.4e-07 |
| eurosat | Full FT | cifar100_retention | 12 | -0.972 | 1.4e-07 |
| eurosat | Full FT | test_acc | 12 | -0.726 | 0.0075 |
| eurosat | LoRA r=8 | delta_entropy_pct | 12 | +0.972 | 1.4e-07 |
| eurosat | LoRA r=8 | cifar100_retention | 12 | -0.972 | 1.4e-07 |
| eurosat | LoRA r=8 | test_acc | 12 | +0.972 | 1.4e-07 |
| pets | Full FT | delta_entropy_pct | 12 | -0.972 | 1.4e-07 |
| pets | Full FT | cifar100_retention | 12 | -0.972 | 1.4e-07 |
| pets | Full FT | test_acc | 12 | +0.324 | 0.3 |
| pets | LoRA r=8 | delta_entropy_pct | 12 | -0.885 | 0.00013 |
| pets | LoRA r=8 | cifar100_retention | 12 | -0.324 | 0.3 |
| pets | LoRA r=8 | test_acc | 12 | +0.972 | 1.4e-07 |

## Predictors of CIFAR-100 retention (final)

| predictor | n | rho overall | p | mean |rho| within dataset | mean |rho| within dataset x method |
|---|---|---|---|---|---|
| epoch1_transfer_retention_pct | 48 | +0.967 | 6.1e-29 | 0.952 | 0.804 |
| cka_mean | 48 | +0.960 | 5.6e-27 | 0.947 | 0.787 |
| embedding_drift | 48 | -0.919 | 3.4e-20 | 0.923 | 0.752 |
| cka_last | 48 | +0.877 | 3e-16 | 0.902 | 0.605 |
| abs_delta_gini_pct | 48 | -0.844 | 4.7e-14 | 0.791 | 0.692 |
| weight_drift_rel | 48 | -0.837 | 1.3e-13 | 0.830 | 0.823 |
| abs_delta_entropy_pct | 48 | -0.836 | 1.4e-13 | 0.792 | 0.656 |
| train_loss_final | 48 | +0.824 | 5.9e-13 | 0.852 | 0.615 |
| abs_delta_erf95_pct | 48 | -0.796 | 1.4e-11 | 0.816 | 0.659 |
| abs_delta_entropy_last_layer_pct | 48 | -0.739 | 2e-09 | 0.833 | 0.663 |
| test_acc | 48 | -0.606 | 5e-06 | 0.606 | 0.559 |
| delta_entropy_last_layer_pct | 48 | +0.431 | 0.0023 | 0.605 | 0.788 |
| log10_lr | 48 | -0.382 | 0.0074 | 0.404 | 0.810 |
| delta_entropy_pct | 48 | +0.361 | 0.012 | 0.505 | 0.860 |

## Predictors of CIFAR-100 retention (epoch1)

| predictor | n | rho overall | p | mean |rho| within dataset | mean |rho| within dataset x method |
|---|---|---|---|---|---|
| epoch1_transfer_retention_pct | 48 | +0.967 | 6.1e-29 | 0.952 | 0.804 |
| cka_mean | 48 | +0.959 | 9e-27 | 0.953 | 0.813 |
| embedding_drift | 48 | -0.941 | 3.4e-23 | 0.941 | 0.827 |
| weight_drift_rel | 48 | -0.906 | 8.6e-19 | 0.933 | 0.839 |
| abs_delta_entropy_pct | 48 | -0.865 | 2.2e-15 | 0.846 | 0.663 |
| abs_delta_entropy_last_layer_pct | 48 | -0.852 | 1.6e-14 | 0.825 | 0.643 |
| abs_delta_gini_pct | 48 | -0.844 | 4.8e-14 | 0.829 | 0.722 |
| abs_delta_erf95_pct | 48 | -0.830 | 3e-13 | 0.803 | 0.760 |
| train_loss_final | 48 | +0.570 | 2.3e-05 | 0.863 | 0.734 |
| log10_lr | 48 | -0.382 | 0.0074 | 0.404 | 0.810 |
| delta_entropy_last_layer_pct | 48 | +0.382 | 0.0074 | 0.480 | 0.818 |
| delta_entropy_pct | 48 | +0.237 | 0.1 | 0.564 | 0.797 |

## Bootstrap 95% CI on |rho| and paired gap vs CKA

| predictor | |rho| | 95% CI | gap vs CKA | gap CI | CKA wins |
|---|---|---|---|---|---|
| cka_mean | 0.968 | [0.835, 1.000] | +0.000 | [+0.000, +0.000] | 0.000 |
| epoch1_transfer_retention_pct | 0.935 | [0.701, 1.000] | +0.035 | [-0.071, +0.204] | 0.680 |
| embedding_drift | 0.929 | [0.750, 0.982] | +0.049 | [-0.036, +0.167] | 0.861 |
| abs_delta_entropy_pct | 0.874 | [0.574, 0.979] | +0.105 | [-0.030, +0.408] | 0.886 |
| abs_delta_gini_pct | 0.865 | [0.587, 0.967] | +0.115 | [-0.018, +0.390] | 0.935 |
| train_loss_final | 0.844 | [0.580, 0.945] | +0.140 | [-0.036, +0.394] | 0.935 |
| weight_drift_rel | 0.835 | [0.540, 0.964] | +0.144 | [-0.018, +0.424] | 0.952 |
| abs_delta_entropy_last_layer_pct | 0.756 | [0.385, 0.926] | +0.226 | [+0.030, +0.573] | 0.999 |
| test_acc | 0.641 | [0.172, 0.858] | +0.340 | [+0.065, +0.796] | 0.997 |
| log10_lr | 0.376 | [0.028, 0.777] | +0.567 | [+0.154, +0.944] | 0.996 |
| delta_entropy_pct | 0.312 | [0.019, 0.765] | +0.601 | [+0.149, +0.948] | 0.997 |

## Best-validation operating point per method

- **eurosat**: Full FT lr=1e-05 test=97.19% C100=41.2% dH=+0.99% | LoRA lr=1e-04 test=95.16% C100=86.4% dH=+0.98%
- **pets**: Full FT lr=3e-05 test=88.26% C100=21.8% dH=-1.90% | LoRA lr=1e-04 test=89.88% C100=97.7% dH=-0.35%

## Signal-based configuration choice

- **eurosat**: 8 candidates above 95.00% validation accuracy; retention range 17.1--58.1%, random pick 41.2%
    - abs_delta_entropy_pct: picks S2_eurosat_full_ft_lr3e-5_seed19 -> retention 24.0% (regret 34.1, range fraction 0.17)
    - abs_delta_entropy_last_layer_pct: picks S2_eurosat_full_ft_lr1e-5_seed19 -> retention 43.6% (regret 14.4, range fraction 0.65)
    - abs_delta_erf95_pct: picks S2_eurosat_full_ft_lr3e-5_seed19 -> retention 24.0% (regret 34.1, range fraction 0.17)
    - abs_delta_gini_pct: picks S2_eurosat_full_ft_lr3e-5_seed7 -> retention 17.1% (regret 40.9, range fraction 0.00)
    - embedding_drift: picks S2_eurosat_full_ft_lr3e-6_seed19 -> retention 56.9% (regret 1.1, range fraction 0.97)
    - weight_drift_rel: picks S2_eurosat_full_ft_lr3e-6_seed19 -> retention 56.9% (regret 1.1, range fraction 0.97)
    - cka_mean: picks S2_eurosat_full_ft_lr3e-6_seed11 -> retention 50.1% (regret 8.0, range fraction 0.80)
- **pets**: 6 candidates above 90.16% validation accuracy; retention range 19.3--99.7%, random pick 50.2%
    - abs_delta_entropy_pct: picks S2_pets_lora_r8_lr1e-4_seed19 -> retention 99.7% (regret 0.0, range fraction 1.00)
    - abs_delta_entropy_last_layer_pct: picks S2_pets_lora_r8_lr1e-4_seed19 -> retention 99.7% (regret 0.0, range fraction 1.00)
    - abs_delta_erf95_pct: picks S2_pets_lora_r8_lr1e-4_seed19 -> retention 99.7% (regret 0.0, range fraction 1.00)
    - abs_delta_gini_pct: picks S2_pets_lora_r8_lr1e-4_seed19 -> retention 99.7% (regret 0.0, range fraction 1.00)
    - embedding_drift: picks S2_pets_lora_r8_lr1e-4_seed7 -> retention 97.0% (regret 2.7, range fraction 0.97)
    - weight_drift_rel: picks S2_pets_full_ft_lr1e-5_seed7 -> retention 38.8% (regret 60.9, range fraction 0.24)
    - cka_mean: picks S2_pets_lora_r8_lr1e-4_seed7 -> retention 97.0% (regret 2.7, range fraction 0.97)

## Accuracy / retention frontier (EuroSAT, all configurations)

| method | lr | n | test acc | C100 ret | w-drift |
|---|---|---|---|---|---|
| LoRA (frozen proj.) | 1e-05 | 3 | 22.63 | 98.6 | 0.0010 |
| LoRA r=8 | 3e-06 | 3 | 45.54 | 99.6 | 0.0003 |
| LoRA r=8 | 1e-05 | 3 | 78.83 | 98.2 | 0.0010 |
| Linear probe | 1e-03 | 3 | 85.11 | 100.0 | 0.0000 |
| Last block | 1e-05 | 3 | 88.57 | 97.1 | 0.0040 |
| LoRA r=8 | 3e-05 | 3 | 91.73 | 94.5 | 0.0024 |
| LoRA (frozen proj.) | 1e-04 | 3 | 92.67 | 76.6 | 0.0063 |
| Full FT | 1e-04 | 3 | 94.02 | 3.7 | 0.0222 |
| Last block | 1e-04 | 3 | 94.38 | 83.8 | 0.0109 |
| LoRA r=8 | 1e-04 | 3 | 95.16 | 86.4 | 0.0042 |
| Full FT | 3e-05 | 3 | 95.98 | 22.3 | 0.0083 |
| Full FT | 3e-06 | 3 | 96.81 | 55.0 | 0.0022 |
| Full FT | 1e-05 | 3 | 97.19 | 41.2 | 0.0042 |

### Pareto-optimal configurations (accuracy vs retention)

- **eurosat**: Full FT @1e-05 (97.2/41.2); Full FT @3e-06 (96.8/55.0); LoRA r=8 @1e-04 (95.2/86.4); LoRA r=8 @3e-05 (91.7/94.5); Last block @1e-05 (88.6/97.1); Linear probe @1e-03 (85.1/100.0)
  dominated: Full FT @3e-05; Full FT @1e-04; Last block @1e-04; LoRA r=8 @3e-06; LoRA r=8 @1e-05; LoRA (frozen proj.) @1e-05; LoRA (frozen proj.) @1e-04
- **pets**: LoRA r=8 @1e-04 (89.9/97.7); Linear probe @1e-03 (80.5/100.0); LoRA r=8 @3e-05 (71.7/100.8)
  dominated: Full FT @3e-06; Full FT @1e-05; Full FT @3e-05; Full FT @1e-04; LoRA r=8 @3e-06; LoRA r=8 @1e-05

## Selection choice, sensitivity to the accuracy band

| dataset | margin | candidates | oracle | random | CKA | emb. drift | w-drift | |dH| |
|---|---|---|---|---|---|---|---|---|
| eurosat | 1 | 5 | 58.1 | 39.7 | 58.1 | 58.1 | 58.1 | 17.1 |
| pets | 1 | 4 | 99.7 | 65.5 | 97.0 | 97.0 | 38.8 | 99.7 |
| eurosat | 2 | 8 | 58.1 | 41.2 | 50.1 | 56.9 | 56.9 | 24.0 |
| pets | 2 | 6 | 99.7 | 50.2 | 97.0 | 97.0 | 38.8 | 99.7 |
| eurosat | 3 | 11 | 87.6 | 40.7 | 87.6 | 87.6 | 56.9 | 24.0 |
| pets | 3 | 9 | 99.7 | 54.6 | 96.4 | 96.4 | 54.4 | 99.7 |
| eurosat | 5 | 15 | 87.6 | 41.7 | 85.5 | 86.2 | 56.9 | 24.0 |
| pets | 5 | 9 | 99.7 | 54.6 | 96.4 | 96.4 | 54.4 | 99.7 |

## Second backbone: `openai/clip-vit-base-patch16` (18 runs)

| dataset | method | lr | n | test acc | dH % | C100 ret % | CKA |
|---|---|---|---|---|---|---|---|
| eurosat | Full FT | 1e-05 | 3 | 97.06 | -0.94 | 52.6 | 0.808 |
| eurosat | Full FT | 3e-05 | 3 | 97.53 | +2.25 | 29.2 | 0.772 |
| eurosat | Full FT | 1e-04 | 3 | 95.65 | +3.24 | 2.3 | 0.743 |
| eurosat | LoRA r=8 | 1e-05 | 3 | 80.15 | +0.43 | 99.9 | 0.972 |
| eurosat | LoRA r=8 | 3e-05 | 3 | 93.21 | +0.64 | 98.2 | 0.928 |
| eurosat | LoRA r=8 | 1e-04 | 3 | 95.49 | +0.68 | 92.8 | 0.903 |

Predictors of retention on this backbone:

| predictor | n | rho | p |
|---|---|---|---|
| cka_mean | 18 | +0.990 | 6.3e-15 |
| embedding_drift | 18 | -0.953 | 1.1e-09 |
| weight_drift_rel | 18 | -0.965 | 1e-10 |
| abs_delta_entropy_pct | 18 | -0.717 | 0.00081 |
| delta_entropy_pct | 18 | -0.472 | 0.048 |

## Polarity by backbone x method (EuroSAT, run level)

| backbone | method | n | rates | signed dH rho | CKA rho | w-drift rho |
|---|---|---|---|---|---|---|
| clip-vit-base-patch16 | full_ft | 9 | 3 | -0.717 | +0.933 | -0.917 |
| clip-vit-base-patch16 | lora_r8 | 9 | 3 | +0.000 | +0.983 | -0.983 |
| clip-vit-base-patch32 | full_ft | 12 | 4 | +0.944 | +0.944 | -0.951 |
| clip-vit-base-patch32 | lora_r8 | 12 | 4 | -0.951 | +0.888 | -0.923 |

## Early warning (epoch 1 signal vs final retention)

```json
{
  "n_runs": 48,
  "epoch1_delta_entropy_pct_vs_final": {
    "spearman_rho": 0.23719062092922275,
    "p_value": 0.1045368360462821
  },
  "epoch1_weight_drift_rel_vs_final": {
    "spearman_rho": -0.9059921841076857,
    "p_value": 8.603581484770055e-19
  },
  "epoch1_cka_mean_vs_final": {
    "spearman_rho": 0.9587494572297004,
    "p_value": 9.038592810290212e-27
  },
  "cell_level": {
    "n_damaged": 6,
    "n_healthy": 10,
    "delta_entropy_pct": {
      "auc_oriented": 0.6833333333333333
    },
    "abs_delta_entropy_pct": {
      "auc_oriented": 0.95
    },
    "weight_drift_rel": {
      "auc_oriented": 0.9833333333333333
    },
    "embedding_drift": {
      "auc_oriented": 1.0
    },
    "cka_mean": {
      "auc_oriented": 1.0
    }
  },
  "epoch1_delta_entropy_pct_auc_for_damage": {
    "auc_oriented": 0.6584440227703985,
    "damaged_scores_lower": true,
    "n_damaged": 17,
    "n_healthy": 31,
    "damaged_mean": -0.6593176779502543,
    "healthy_mean": 0.1384178936410597
  },
  "epoch1_abs_delta_entropy_pct_auc_for_damage": {
    "auc_oriented": 0.9392789373814042,
    "damaged_scores_lower": false,
    "n_damaged": 17,
    "n_healthy": 31,
    "damaged_mean": 1.4208404087390811,
    "healthy_mean": 0.17705688283964793
  },
  "epoch1_weight_drift_rel_auc_for_damage": {
    "auc_oriented": 0.9886148007590133,
    "damaged_scores_lower": false,
    "n_damaged": 17,
    "n_healthy": 31,
    "damaged_mean": 0.007842816214273516,
    "healthy_mean": 0.0007770473998162503
  },
  "epoch1_embedding_drift_auc_for_damage": {
    "auc_oriented": 1.0,
    "damaged_scores_lower": false,
    "n_damaged": 17,
    "n_healthy": 31,
    "damaged_mean": 0.6050789286108578,
    "healthy_mean": 0.10645413014196581
  },
  "epoch1_cka_mean_auc_for_damage": {
    "auc_oriented": 0.9886148007590133,
    "damaged_scores_lower": true,
    "n_damaged": 17,
    "n_healthy": 31,
    "damaged_mean": 0.7034392313045614,
    "healthy_mean": 0.960774018399177
  }
}
```
