# Paper numbers, in document order

Each row is a directive from `main_template.tex` and the value it
resolved to. Cross-check against `digest.md`.

| # | context | directive | value |
|---|---|---|---|
| 1 | ...uted (48 matched runs plus reference configurations, | `@nruns()@` | 66 |
| 2 | ...h{broadens} attention at the smallest matched rate ( | `@ms(eurosat,full_ft,3e-6,delta_entropy_pct)@` | 1.53\,$\pm$\,0.25 |
| 3 | ...opy_pct)@\%), is neutral in the middle of the grid ( | `@cell(eurosat,full_ft,3e-5,delta_entropy_pct,+.2f)@` | +0.06 |
| 4 | ...entropy_pct,+.2f)@\%) and contracts at the largest ( | `@cell(eurosat,full_ft,1e-4,delta_entropy_pct,+.2f)@` | -2.34 |
| 5 | ...; the trend is monotone in the rate (Spearman $\rho= | `@dose(eurosat,full_ft,delta_entropy_pct)@` | -0.972 |
| 6 | ...\rho=@dose(eurosat,full_ft,delta_entropy_pct)@$, $p= | `@dose(eurosat,full_ft,delta_entropy_pct,p_value,sci)@` | 1.4\!\cdot\!10^{-7} |
| 7 | ...ts entropy \emph{rises} with the learning rate, from | `@cell(eurosat,lora_r8,3e-6,delta_entropy_pct,+.2f)@` | +0.01 |
| 8 | ...l(eurosat,lora_r8,3e-6,delta_entropy_pct,+.2f)@\% to | `@cell(eurosat,lora_r8,1e-4,delta_entropy_pct,+.2f)@` | +0.98 |
| 9 | ...osat,lora_r8,1e-4,delta_entropy_pct,+.2f)@\% ($\rho= | `@dose(eurosat,lora_r8,delta_entropy_pct)@` | 0.972 |
| 10 | ...s is more contractive throughout: Full FT falls from | `@cell(pets,full_ft,3e-6,delta_entropy_pct,+.2f)@` | -0.62 |
| 11 | ...cell(pets,full_ft,3e-6,delta_entropy_pct,+.2f)@\% to | `@cell(pets,full_ft,1e-4,delta_entropy_pct,+.2f)@` | -3.23 |
| 12 | ...ull_ft,1e-4,delta_entropy_pct,+.2f)@\% and LoRA from | `@cell(pets,lora_r8,3e-6,delta_entropy_pct,+.2f)@` | -0.00 |
| 13 | ...cell(pets,lora_r8,3e-6,delta_entropy_pct,+.2f)@\% to | `@cell(pets,lora_r8,1e-4,delta_entropy_pct,+.2f)@` | -0.35 |
| 14 | ...ng rate in three of the four groups (Spearman $\rho= | `@dosebest(cifar100_retention)@` | -0.972 |
| 15 | ...$\rho=@dosebest(cifar100_retention)@$ in each), from | `@cell(eurosat,full_ft,3e-6,cifar100_retention,.1f)@` | 55.0 |
| 16 | ...l(eurosat,full_ft,3e-6,cifar100_retention,.1f)@\% to | `@cell(eurosat,full_ft,1e-4,cifar100_retention,.1f)@` | 3.7 |
| 17 | ...ar100_retention,.1f)@\% for EuroSAT Full FT and from | `@cell(eurosat,lora_r8,3e-6,cifar100_retention,.1f)@` | 99.6 |
| 18 | ...l(eurosat,lora_r8,3e-6,cifar100_retention,.1f)@\% to | `@cell(eurosat,lora_r8,1e-4,cifar100_retention,.1f)@` | 86.4 |
| 19 | ...hat it forgets almost nothing anywhere in the grid ( | `@cell(pets,lora_r8,3e-6,cifar100_retention,.1f)@` | 100.0 |
| 20 | ...cell(pets,lora_r8,3e-6,cifar100_retention,.1f)@\% to | `@cell(pets,lora_r8,1e-4,cifar100_retention,.1f)@` | 97.7 |
| 21 | ...pets,lora_r8,1e-4,cifar100_retention,.1f)@\%, $\rho= | `@dose(pets,lora_r8,cifar100_retention)@` | -0.324 |
| 22 | ... $\rho=@dose(pets,lora_r8,cifar100_retention)@$, $p= | `@dose(pets,lora_r8,cifar100_retention,p_value,.2f)@` | 0.30 |
| 23 | ...er identical (learning rate, seed) cells, LoRA keeps | `@paired(eurosat,cifar100_retention,mean_difference,abs1)@` | 64.2 |
| 24 | ...ined CIFAR-100 accuracy than Full FT on EuroSAT ($p= | `@paired(eurosat,cifar100_retention,t_p_holm,sci)@` | 2.0\!\cdot\!10^{-7} |
| 25 | ...ntion,t_p_holm,sci)@$ after Holm correction, $\|d_z\|= | `@paired(eurosat,cifar100_retention,cohens_dz,abs2)@` | 4.14 |
| 26 | ...ed(eurosat,cifar100_retention,cohens_dz,abs2)@$) and | `@paired(pets,cifar100_retention,mean_difference,abs1)@` | 62.5 |
| 27 | ...tion,mean_difference,abs1)@ points more on Pets ($p= | `@paired(pets,cifar100_retention,t_p_holm,sci)@` | 2.9\!\cdot\!10^{-5} |
| 28 | ...to the pretrained representation (mean CKA higher by | `@paired(eurosat,cka_mean,mean_difference,abs3)@` | 0.164 |
| 29 | ...@paired(eurosat,cka_mean,mean_difference,abs3)@, $p= | `@paired(eurosat,cka_mean,t_p_holm,sci)@` | 3.4\!\cdot\!10^{-12} |
| 30 | ...t: on EuroSAT the paired difference in $\Delta H$ is | `@paired(eurosat,delta_entropy_pct,mean_difference,abs2)@` | 0.41 |
| 31 | ...lta_entropy_pct,mean_difference,abs2)@ points at $p= | `@paired(eurosat,delta_entropy_pct,t_p_value,.2f)@` | 0.49 |
| 32 | ...ta_entropy_pct,t_p_value,.2f)@$, while on Pets it is | `@paired(pets,delta_entropy_pct,mean_difference,abs2)@` | 1.63 |
| 33 | ...lta_entropy_pct,mean_difference,abs2)@ points at $p= | `@paired(pets,delta_entropy_pct,t_p_holm,sci)@` | 3.2\!\cdot\!10^{-4} |
| 34 | ...rk rather than subtle. On EuroSAT, Full FT selects $ | `@iso(eurosat,full_ft_lr,.0e)@` | 10^{-5} |
| 35 | ... selects $@iso(eurosat,full_ft_lr,.0e)@$ and reaches | `@iso(eurosat,full_ft_test_acc,.2f)@` | 97.19 |
| 36 | ...(eurosat,full_ft_test_acc,.2f)@\% test accuracy with | `@iso(eurosat,full_ft_cifar100_retention,.1f)@` | 41.2 |
| 37 | ...00_retention,.1f)@\% retention, while LoRA selects $ | `@iso(eurosat,lora_r8_lr,.0e)@` | 10^{-4} |
| 38 | ... selects $@iso(eurosat,lora_r8_lr,.0e)@$ and reaches | `@iso(eurosat,lora_r8_test_acc,.2f)@` | 95.16 |
| 39 | ...d reaches @iso(eurosat,lora_r8_test_acc,.2f)@\% with | `@iso(eurosat,lora_r8_cifar100_retention,.1f)@` | 86.4 |
| 40 | ...1f)@\%. On Pets, LoRA is better on \emph{both} axes: | `@iso(pets,lora_r8_test_acc,.2f)@` | 89.88 |
| 41 | ...oth} axes: @iso(pets,lora_r8_test_acc,.2f)@\% versus | `@iso(pets,full_ft_test_acc,.2f)@` | 88.26 |
| 42 | ...so(pets,full_ft_test_acc,.2f)@\% target accuracy and | `@iso(pets,lora_r8_cifar100_retention,.1f)@` | 97.7 |
| 43 | ... @iso(pets,lora_r8_cifar100_retention,.1f)@\% versus | `@iso(pets,full_ft_cifar100_retention,.1f)@` | 21.8 |
| 44 | ...& $\Delta H_{12}$ (\%) & C100 ret.\ (\%) \\ \midrule | `@rows()@` | EuroSAT & Full FT & $3\!\cdot\!10^{-6}$ & 3 & 96.81\,$\pm$\,0.07 & 1.53\,$\pm$\,0.25 & 3.39\,$\pm$\,0.38 & 55.03\,$\pm$\,4.33 \\
EuroSAT & Full FT & $10^{-5}$ & 3 & 97.19\,$\pm$\,0.33 & 0.99\,$\pm$\,0.17 & 1.24\,$\pm$\,0.36 & 41.15\,$\pm$\,2.31 \\
EuroSAT & Full FT & $3\!\cdot\!10^{-5}$ & 3 & 95.98\,$\pm$\,1.36 & 0.06\,$\pm$\,0.21 & -2.39\,$\pm$\,0.48 & 22.27\,$\pm$\,4.52 \\
EuroSAT & Full FT & $10^{-4}$ & 3 & 94.02\,$\pm$\,0.43 & -2.34\,$\pm$\,0.40 & -7.75\,$\pm$\,2.87 & 3.65\,$\pm$\,0.62 \\
EuroSAT & LoRA r=8 & $3\!\cdot\!10^{-6}$ & 3 & 45.54\,$\pm$\,0.80 & 0.01\,$\pm$\,0.02 & 0.00\,$\pm$\,0.08 & 99.65\,$\pm$\,0.42 \\
EuroSAT & LoRA r=8 & $10^{-5}$ & 3 & 78.83\,$\pm$\,1.54 & 0.16\,$\pm$\,0.07 & 0.37\,$\pm$\,0.33 & 98.18\,$\pm$\,0.69 \\
EuroSAT & LoRA r=8 & $3\!\cdot\!10^{-5}$ & 3 & 91.73\,$\pm$\,0.19 & 0.71\,$\pm$\,0.08 & 2.04\,$\pm$\,0.38 & 94.52\,$\pm$\,0.41 \\
EuroSAT & LoRA r=8 & $10^{-4}$ & 3 & 95.16\,$\pm$\,0.44 & 0.98\,$\pm$\,0.03 & 1.71\,$\pm$\,0.27 & 86.44\,$\pm$\,1.09 \\
Oxford-IIIT Pets & Full FT & $3\!\cdot\!10^{-6}$ & 3 & 75.82\,$\pm$\,2.47 & -0.62\,$\pm$\,0.26 & -4.95\,$\pm$\,0.38 & 73.54\,$\pm$\,5.71 \\
Oxford-IIIT Pets & Full FT & $10^{-5}$ & 3 & 87.72\,$\pm$\,0.20 & -1.23\,$\pm$\,0.26 & -6.97\,$\pm$\,0.48 & 44.17\,$\pm$\,8.87 \\
Oxford-IIIT Pets & Full FT & $3\!\cdot\!10^{-5}$ & 3 & 88.26\,$\pm$\,0.26 & -1.90\,$\pm$\,0.22 & -10.30\,$\pm$\,0.62 & 21.83\,$\pm$\,3.98 \\
Oxford-IIIT Pets & Full FT & $10^{-4}$ & 3 & 79.10\,$\pm$\,0.49 & -3.23\,$\pm$\,0.20 & -18.51\,$\pm$\,1.29 & 9.16\,$\pm$\,1.96 \\
Oxford-IIIT Pets & LoRA r=8 & $3\!\cdot\!10^{-6}$ & 3 & 6.23\,$\pm$\,0.93 & -0.00\,$\pm$\,0.00 & -0.01\,$\pm$\,0.02 & 99.98\,$\pm$\,0.27 \\
Oxford-IIIT Pets & LoRA r=8 & $10^{-5}$ & 3 & 26.56\,$\pm$\,1.50 & -0.00\,$\pm$\,0.01 & -0.08\,$\pm$\,0.10 & 100.11\,$\pm$\,0.62 \\
Oxford-IIIT Pets & LoRA r=8 & $3\!\cdot\!10^{-5}$ & 3 & 71.75\,$\pm$\,1.42 & -0.11\,$\pm$\,0.06 & -2.03\,$\pm$\,0.47 & 100.82\,$\pm$\,1.77 \\
Oxford-IIIT Pets & LoRA r=8 & $10^{-4}$ & 3 & 89.88\,$\pm$\,0.47 & -0.35\,$\pm$\,0.08 & -5.94\,$\pm$\,0.44 & 97.70\,$\pm$\,1.78 \\ |
| 45 | ...aluation set itself, read after one epoch, at $\rho= | `@pred(epoch1_transfer_retention_pct,spearman_overall)@` | 0.967 |
| 46 | ...all)@$. Layerwise CKA comes closest without labels ( | `@pred(cka_mean,spearman_overall)@` | 0.960 |
| 47 | ...cka_mean,spearman_overall)@), then embedding drift ( | `@pred(embedding_drift,spearman_overall)@` | -0.919 |
| 48 | ...arman_overall)@), then a group of $\|\Delta$Gini$\|$ ( | `@pred(abs_delta_gini_pct,spearman_overall)@` | -0.844 |
| 49 | ...ni_pct,spearman_overall)@), relative weight change ( | `@pred(weight_drift_rel,spearman_overall)@` | -0.837 |
| 50 | ...ght_drift_rel,spearman_overall)@) and $\|\Delta H\|$ ( | `@pred(abs_delta_entropy_pct,spearman_overall)@` | -0.836 |
| 51 | ...rget-task accuracy, which needs labels, manages only | `@pred(test_acc,spearman_overall)@` | -0.606 |
| 52 | ... --- the quantity our earlier work reported --- only | `@pred(delta_entropy_pct,spearman_overall)@` | 0.361 |
| 53 | ...t evidence and shrink every interval; resampling the | `@boot(cka_mean,n_units,.0f)@` | 16 |
| 54 | ... advantage excludes zero against signed $\Delta H$ ( | `@boot(delta_entropy_pct,gap_vs_reference_mean,+.3f)@` | +0.601 |
| 55 | ..._entropy_pct,gap_vs_reference_mean,+.3f)@, 95\% CI [ | `@boot(delta_entropy_pct,gap_ci_low,.3f)@` | 0.149 |
| 56 | ..., 95\% CI [@boot(delta_entropy_pct,gap_ci_low,.3f)@, | `@boot(delta_entropy_pct,gap_ci_high,.3f)@` | 0.948 |
| 57 | ...ropy_pct,gap_ci_high,.3f)@]) and $\|\Delta H_{12}\|$ ( | `@boot(abs_delta_entropy_last_layer_pct,gap_vs_reference_mean,+.3f)@` | +0.226 |
| 58 | ...ntropy_last_layer_pct,gap_vs_reference_mean,+.3f)@ [ | `@boot(abs_delta_entropy_last_layer_pct,gap_ci_low,.3f)@` | 0.030 |
| 59 | ...t(abs_delta_entropy_last_layer_pct,gap_ci_low,.3f)@, | `@boot(abs_delta_entropy_last_layer_pct,gap_ci_high,.3f)@` | 0.573 |
| 60 | ...t embedding drift, $\|\Delta H\|$ or the weight norm ( | `@boot(weight_drift_rel,gap_vs_reference_mean,+.3f)@` | +0.144 |
| 61 | ...boot(weight_drift_rel,gap_vs_reference_mean,+.3f)@ [ | `@boot(weight_drift_rel,gap_ci_low,.3f)@` | -0.018 |
| 62 | ...ean,+.3f)@ [@boot(weight_drift_rel,gap_ci_low,.3f)@, | `@boot(weight_drift_rel,gap_ci_high,.3f)@` | 0.424 |
| 63 | ...hange is the best signal in the study (mean $\|\rho\|$ | `@pred(delta_entropy_pct,abs_spearman_within_mean)@` | 0.860 |
| 64 | ...)@ over the four groups), ahead of the weight norm ( | `@pred(weight_drift_rel,abs_spearman_within_mean)@` | 0.823 |
| 65 | ...earman_within_mean)@), the log learning rate alone ( | `@pred(log10_lr,abs_spearman_within_mean)@` | 0.810 |
| 66 | ...e (@pred(log10_lr,abs_spearman_within_mean)@), CKA ( | `@pred(cka_mean,abs_spearman_within_mean)@` | 0.787 |
| 67 | ...,abs_spearman_within_mean)@) and its own magnitude ( | `@pred(abs_delta_entropy_pct,abs_spearman_within_mean)@` | 0.656 |
| 68 | ...gn} is positive only for ViT-B/32 full fine-tuning ( | `@pol(patch32,full_ft,delta_entropy_pct)@` | +0.94 |
| 69 | ...ta_entropy_pct)@) and negative for the other three ( | `@pol(patch32,lora_r8,delta_entropy_pct)@` | -0.95 |
| 70 | ...her three (@pol(patch32,lora_r8,delta_entropy_pct)@, | `@pol(patch16,full_ft,delta_entropy_pct)@` | -0.80 |
| 71 | ...opy_pct)@, @pol(patch16,full_ft,delta_entropy_pct)@, | `@pol(patch16,lora_r8,delta_entropy_pct)@` | -1.00 |
| 72 | ..._entropy_pct)@), where CKA is positive in all four ( | `@polrange(cka_mean,min)@` | +0.89 |
| 73 | ...is positive in all four (@polrange(cka_mean,min)@ to | `@polrange(cka_mean,max)@` | +1.00 |
| 74 | ...re alone. The cost shows up in a decision. Among the | `@sel(eurosat,-,n_candidates,.0f)@` | 8 |
| 75 | ...nts of the best \emph{validation} accuracy, spanning | `@sel(eurosat,-,worst_retention)@` | 17.1 |
| 76 | ...ccuracy, spanning @sel(eurosat,-,worst_retention)@-- | `@sel(eurosat,-,oracle_retention)@` | 58.1 |
| 77 | ...--@sel(eurosat,-,oracle_retention)@\% retention with | `@sel(eurosat,-,random_choice_retention)@` | 41.2 |
| 78 | ...andom pick, embedding drift and the weight norm pick | `@sel(eurosat,embedding_drift,retention)@` | 56.9 |
| 79 | ...pick @sel(eurosat,embedding_drift,retention)@\%, CKA | `@sel(eurosat,cka_mean,retention)@` | 50.1 |
| 80 | ...an,retention)@\%, and the smallest $\|\Delta H\|$ only | `@sel(eurosat,abs_delta_entropy_pct,retention)@` | 24.0 |
| 81 | ...s the same rule makes $\|\Delta H\|$ the best choice ( | `@sel(pets,abs_delta_entropy_pct,retention)@` | 99.7 |
| 82 | ...nding at or near the oracle in all eight (worst case | `@selm(2,eurosat,cka_mean,retention)@` | 50.1 |
| 83 | ...,eurosat,cka_mean,retention)@\% against an oracle of | `@selm(2,eurosat,-,oracle_retention)@` | 58.1 |
| 84 | ...very EuroSAT band, and the weight norm never exceeds | `@selm(1,pets,weight_drift_rel,retention)@` | 38.8 |
| 85 | ...1,pets,weight_drift_rel,retention)@\% on Pets, where | `@selm(1,pets,-,oracle_retention)@` | 99.7 |
| 86 | ...wn after the first of six epochs, CKA reaches $\rho= | `@pred1(cka_mean,spearman_overall)@` | 0.959 |
| 87 | ...1(cka_mean,spearman_overall)@$ and the weight norm $ | `@pred1(weight_drift_rel,spearman_overall)@` | -0.906 |
| 88 | ...gainst \emph{final} retention. Asked to separate the | `@stat(temporal_ordering.cell_level.n_damaged,.0f)@` | 6 |
| 89 | ...onfigurations that end below 50\% retention from the | `@stat(temporal_ordering.cell_level.n_healthy,.0f)@` | 10 |
| 90 | ...--- embedding drift and CKA are already perfect (AUC | `@stat(temporal_ordering.cell_level.embedding_drift.auc_oriented,.2f)@` | 1.00 |
| 91 | ...ift.auc_oriented,.2f)@), the weight norm nearly so ( | `@stat(temporal_ordering.cell_level.weight_drift_rel.auc_oriented,.2f)@` | 0.98 |
| 92 | ..._oriented,.2f)@), and signed drift close to chance ( | `@stat(temporal_ordering.cell_level.delta_entropy_pct.auc_oriented,.2f)@` | 0.68 |
| 93 | ... last block under Full FT and growing with the rate, | `@cell(eurosat,full_ft,1e-4,delta_entropy_last_layer_pct,+.1f)@` | -7.7 |
| 94 | ...delta_entropy_last_layer_pct,+.1f)@\% on EuroSAT and | `@cell(pets,full_ft,1e-4,delta_entropy_last_layer_pct,+.1f)@` | -18.5 |
| 95 | ...layer_pct,+.1f)@\% on Pets against block averages of | `@cell(eurosat,full_ft,1e-4,delta_entropy_pct,+.1f)@` | -2.3 |
| 96 | ...(eurosat,full_ft,1e-4,delta_entropy_pct,+.1f)@\% and | `@cell(pets,full_ft,1e-4,delta_entropy_pct,+.1f)@` | -3.2 |
| 97 | ...drift and full retention, and it does, at $\Delta H= | `@cell(eurosat,linear_probe,1e-3,delta_entropy_pct,+.2f)@` | +0.00 |
| 98 | ...at,linear_probe,1e-3,delta_entropy_pct,+.2f)@\%$ and | `@cell(eurosat,linear_probe,1e-3,cifar100_retention,.1f)@` | 100.0 |
| 99 | ...sat,linear_probe,1e-3,cifar100_retention,.1f)@\% for | `@cell(eurosat,linear_probe,1e-3,test_acc,.1f)@` | 85.1 |
| 100 | ... span all four families, from Full FT at $10^{-5}$ ( | `@cell(eurosat,full_ft,1e-5,test_acc,.1f)@` | 97.2 |
| 101 | ...10^{-5}$ (@cell(eurosat,full_ft,1e-5,test_acc,.1f)@/ | `@cell(eurosat,full_ft,1e-5,cifar100_retention,.1f)@` | 41.2 |
| 102 | ...ifar100_retention,.1f)@) through LoRA at $10^{-4}$ ( | `@cell(eurosat,lora_r8,1e-4,test_acc,.1f)@` | 95.2 |
| 103 | ...10^{-4}$ (@cell(eurosat,lora_r8,1e-4,test_acc,.1f)@/ | `@cell(eurosat,lora_r8,1e-4,cifar100_retention,.1f)@` | 86.4 |
| 104 | ...ar100_retention,.1f)@) and last-block at $10^{-5}$ ( | `@cell(eurosat,last_block,1e-5,test_acc,.1f)@` | 88.6 |
| 105 | ...{-5}$ (@cell(eurosat,last_block,1e-5,test_acc,.1f)@/ | `@cell(eurosat,last_block,1e-5,cifar100_retention,.1f)@` | 97.1 |
| 106 | ...A at $10^{-4}$ beats all four on both axes at once ( | `@cell(pets,lora_r8,1e-4,test_acc,.1f)@` | 89.9 |
| 107 | ...xes at once (@cell(pets,lora_r8,1e-4,test_acc,.1f)@/ | `@cell(pets,lora_r8,1e-4,cifar100_retention,.1f)@` | 97.7 |
| 108 | ...ora_r8,1e-4,cifar100_retention,.1f)@ against at best | `@cell(pets,full_ft,3e-5,test_acc,.1f)@` | 88.3 |
| 109 | ...inst at best @cell(pets,full_ft,3e-5,test_acc,.1f)@/ | `@cell(pets,full_ft,3e-5,cifar100_retention,.1f)@` | 21.8 |
| 110 | ...encoder weight than Full FT at $3\!\cdot\!10^{-5}$ ( | `@cell(eurosat,last_block,1e-4,weight_drift_rel,.4f)@` | 0.0109 |
| 111 | ...rosat,last_block,1e-4,weight_drift_rel,.4f)@ against | `@cell(eurosat,full_ft,3e-5,weight_drift_rel,.4f)@` | 0.0083 |
| 112 | ...sat,full_ft,3e-5,weight_drift_rel,.4f)@) yet retains | `@cell(eurosat,last_block,1e-4,cifar100_retention,.1f)@` | 83.8 |
| 113 | ...t,last_block,1e-4,cifar100_retention,.1f)@\% against | `@cell(eurosat,full_ft,3e-5,cifar100_retention,.1f)@` | 22.3 |
| 114 | ...ers and the classifier makes \emph{both} axes worse: | `@cell(eurosat,lora_r8_frozen_proj,1e-4,cifar100_retention,.1f)@` | 76.6 |
| 115 | ...en_proj,1e-4,cifar100_retention,.1f)@\% retention at | `@cell(eurosat,lora_r8_frozen_proj,1e-4,test_acc,.1f)@` | 92.7 |
| 116 | ..._frozen_proj,1e-4,test_acc,.1f)@\% accuracy, against | `@cell(eurosat,lora_r8,1e-4,cifar100_retention,.1f)@` | 86.4 |
| 117 | ...(eurosat,lora_r8,1e-4,cifar100_retention,.1f)@\% and | `@cell(eurosat,lora_r8,1e-4,test_acc,.1f)@` | 95.2 |
| 118 | ...the same rate, with encoder weight drift rising from | `@cell(eurosat,lora_r8,1e-4,weight_drift_rel,.4f)@` | 0.0042 |
| 119 | ...@cell(eurosat,lora_r8,1e-4,weight_drift_rel,.4f)@ to | `@cell(eurosat,lora_r8_frozen_proj,1e-4,weight_drift_rel,.4f)@` | 0.0063 |
| 120 | ...ction leaves encoder drift and retention untouched ( | `@cell(eurosat,lora_r8_frozen_proj,1e-5,weight_drift_rel,.4f)@` | 0.0010 |
| 121 | ...,lora_r8_frozen_proj,1e-5,weight_drift_rel,.4f)@ and | `@cell(eurosat,lora_r8_frozen_proj,1e-5,cifar100_retention,.1f)@` | 98.6 |
| 122 | ..._frozen_proj,1e-5,cifar100_retention,.1f)@\% against | `@cell(eurosat,lora_r8,1e-5,weight_drift_rel,.4f)@` | 0.0010 |
| 123 | ...cell(eurosat,lora_r8,1e-5,weight_drift_rel,.4f)@ and | `@cell(eurosat,lora_r8,1e-5,cifar100_retention,.1f)@` | 98.2 |
| 124 | ...,cifar100_retention,.1f)@\%) but drops accuracy from | `@cell(eurosat,lora_r8,1e-5,test_acc,.1f)@` | 78.8 |
| 125 | ... from @cell(eurosat,lora_r8,1e-5,test_acc,.1f)@\% to | `@cell(eurosat,lora_r8_frozen_proj,1e-5,test_acc,.1f)@` | 22.6 |
| 126 | ... the decision test: at a three-point band it chose a | `@selm(5,pets,weight_drift_rel,retention)@` | 54.4 |
| 127 | ..._drift_rel,retention)@\% configuration on Pets where | `@selm(5,pets,-,oracle_retention)@` | 99.7 |
| 128 | ...er than any other signal we measured, though by only | `@pred(delta_entropy_pct,abs_spearman_within_mean)@` | 0.860 |
| 129 | ...delta_entropy_pct,abs_spearman_within_mean)@ against | `@pred(log10_lr,abs_spearman_within_mean)@` | 0.810 |
