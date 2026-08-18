# Paper numbers, in document order

Each row is a directive from `main_template.tex` and the value it
resolved to. Cross-check against `digest.md`.

| # | context | directive | value |
|---|---|---|---|
| 1 | ...relate of forgetting (mean absolute rank correlation | `@pred(delta_entropy_pct,abs_spearman_within_mean)@` | 0.860 |
| 2 | ...oups and its cross-configuration correlation is only | `@pred(delta_entropy_pct,spearman_overall)@` | 0.361 |
| 3 | ...ge-free, auditable dose-response analysis comprising | `@nruns()@` | 66 |
| 4 | ... using $32\!\times\!32$-pixel patches (ViT-B/32) and | `@bbruns()@` | 18 |
| 5 | ...e EuroSAT grid at three learning rates on ViT-B/16 ( | `@bbruns()@` | 18 |
| 6 | ...tively. The two rank runs almost identically ($\rho= | `@stat(transfer_axis_agreement.spearman_rho,.3f)@` | 0.971 |
| 7 | ...tat(transfer_axis_agreement.spearman_rho,.3f)@$, $n= | `@stat(transfer_axis_agreement.n,.0f)@` | 48 |
| 8 | ...-100; CIFAR-10 ranks runs almost identically, $\rho= | `@stat(transfer_axis_agreement.spearman_rho,.3f)@` | 0.971 |
| 9 | ... H_{12}$ (\%) & CIFAR-100 retention (\%) \\ \midrule | `@rows()@` | EuroSAT & Full FT & $3\cdot 10^{-6}$ & 3 & 96.81\,$\pm$\,0.07 & 1.53\,$\pm$\,0.25 & 3.39\,$\pm$\,0.38 & 55.03\,$\pm$\,4.33 \\
EuroSAT & Full FT & $10^{-5}$ & 3 & 97.19\,$\pm$\,0.33 & 0.99\,$\pm$\,0.17 & 1.24\,$\pm$\,0.36 & 41.15\,$\pm$\,2.31 \\
EuroSAT & Full FT & $3\cdot 10^{-5}$ & 3 & 95.98\,$\pm$\,1.36 & 0.06\,$\pm$\,0.21 & -2.39\,$\pm$\,0.48 & 22.27\,$\pm$\,4.52 \\
EuroSAT & Full FT & $10^{-4}$ & 3 & 94.02\,$\pm$\,0.43 & -2.34\,$\pm$\,0.40 & -7.75\,$\pm$\,2.87 & 3.65\,$\pm$\,0.62 \\
EuroSAT & LoRA r=8 & $3\cdot 10^{-6}$ & 3 & 45.54\,$\pm$\,0.80 & 0.01\,$\pm$\,0.02 & 0.00\,$\pm$\,0.08 & 99.65\,$\pm$\,0.42 \\
EuroSAT & LoRA r=8 & $10^{-5}$ & 3 & 78.83\,$\pm$\,1.54 & 0.16\,$\pm$\,0.07 & 0.37\,$\pm$\,0.33 & 98.18\,$\pm$\,0.69 \\
EuroSAT & LoRA r=8 & $3\cdot 10^{-5}$ & 3 & 91.73\,$\pm$\,0.19 & 0.71\,$\pm$\,0.08 & 2.04\,$\pm$\,0.38 & 94.52\,$\pm$\,0.41 \\
EuroSAT & LoRA r=8 & $10^{-4}$ & 3 & 95.16\,$\pm$\,0.44 & 0.98\,$\pm$\,0.03 & 1.71\,$\pm$\,0.27 & 86.44\,$\pm$\,1.09 \\
Oxford-IIIT Pets & Full FT & $3\cdot 10^{-6}$ & 3 & 75.82\,$\pm$\,2.47 & -0.62\,$\pm$\,0.26 & -4.95\,$\pm$\,0.38 & 73.54\,$\pm$\,5.71 \\
Oxford-IIIT Pets & Full FT & $10^{-5}$ & 3 & 87.72\,$\pm$\,0.20 & -1.23\,$\pm$\,0.26 & -6.97\,$\pm$\,0.48 & 44.17\,$\pm$\,8.87 \\
Oxford-IIIT Pets & Full FT & $3\cdot 10^{-5}$ & 3 & 88.26\,$\pm$\,0.26 & -1.90\,$\pm$\,0.22 & -10.30\,$\pm$\,0.62 & 21.83\,$\pm$\,3.98 \\
Oxford-IIIT Pets & Full FT & $10^{-4}$ & 3 & 79.10\,$\pm$\,0.49 & -3.23\,$\pm$\,0.20 & -18.51\,$\pm$\,1.29 & 9.16\,$\pm$\,1.96 \\
Oxford-IIIT Pets & LoRA r=8 & $3\cdot 10^{-6}$ & 3 & 6.23\,$\pm$\,0.93 & -0.00\,$\pm$\,0.00 & -0.01\,$\pm$\,0.02 & 99.98\,$\pm$\,0.27 \\
Oxford-IIIT Pets & LoRA r=8 & $10^{-5}$ & 3 & 26.56\,$\pm$\,1.50 & -0.00\,$\pm$\,0.01 & -0.08\,$\pm$\,0.10 & 100.11\,$\pm$\,0.62 \\
Oxford-IIIT Pets & LoRA r=8 & $3\cdot 10^{-5}$ & 3 & 71.75\,$\pm$\,1.42 & -0.11\,$\pm$\,0.06 & -2.03\,$\pm$\,0.47 & 100.82\,$\pm$\,1.77 \\
Oxford-IIIT Pets & LoRA r=8 & $10^{-4}$ & 3 & 89.88\,$\pm$\,0.47 & -0.35\,$\pm$\,0.08 & -5.94\,$\pm$\,0.44 & 97.70\,$\pm$\,1.78 \\ |
| 10 | ...h{broadens} attention at the smallest matched rate ( | `@ms(eurosat,full_ft,3e-6,delta_entropy_pct)@` | 1.53\,$\pm$\,0.25 |
| 11 | ...@\%), is approximately neutral at $3\cdot 10^{-5}$ ( | `@cell(eurosat,full_ft,3e-5,delta_entropy_pct,+.2f)@` | +0.06 |
| 12 | ..._entropy_pct,+.2f)@\%), and contracts at $10^{-4}$ ( | `@cell(eurosat,full_ft,1e-4,delta_entropy_pct,+.2f)@` | -2.34 |
| 13 | ...e rate-dependent trend is monotonic (Spearman $\rho= | `@dose(eurosat,full_ft,delta_entropy_pct)@` | -0.972 |
| 14 | ...\rho=@dose(eurosat,full_ft,delta_entropy_pct)@$; $p= | `@dose(eurosat,full_ft,delta_entropy_pct,p_value,sci)@` | 1.4\cdot 10^{-7} |
| 15 | ...its the opposite trend, with entropy increasing from | `@cell(eurosat,lora_r8,3e-6,delta_entropy_pct,+.2f)@` | +0.01 |
| 16 | ...l(eurosat,lora_r8,3e-6,delta_entropy_pct,+.2f)@\% to | `@cell(eurosat,lora_r8,1e-4,delta_entropy_pct,+.2f)@` | +0.98 |
| 17 | ...osat,lora_r8,1e-4,delta_entropy_pct,+.2f)@\% ($\rho= | `@dose(eurosat,lora_r8,delta_entropy_pct)@` | 0.972 |
| 18 | ..., both methods are contractive: Full FT changes from | `@cell(pets,full_ft,3e-6,delta_entropy_pct,+.2f)@` | -0.62 |
| 19 | ...cell(pets,full_ft,3e-6,delta_entropy_pct,+.2f)@\% to | `@cell(pets,full_ft,1e-4,delta_entropy_pct,+.2f)@` | -3.23 |
| 20 | ...ll_ft,1e-4,delta_entropy_pct,+.2f)@\%, and LoRA from | `@cell(pets,lora_r8,3e-6,delta_entropy_pct,+.2f)@` | -0.00 |
| 21 | ...cell(pets,lora_r8,3e-6,delta_entropy_pct,+.2f)@\% to | `@cell(pets,lora_r8,1e-4,delta_entropy_pct,+.2f)@` | -0.35 |
| 22 | ...ng rate in three of the four groups (Spearman $\rho= | `@dosebest(cifar100_retention)@` | -0.972 |
| 23 | ...$\rho=@dosebest(cifar100_retention)@$ in each): from | `@cell(eurosat,full_ft,3e-6,cifar100_retention,.1f)@` | 55.0 |
| 24 | ...l(eurosat,full_ft,3e-6,cifar100_retention,.1f)@\% to | `@cell(eurosat,full_ft,1e-4,cifar100_retention,.1f)@` | 3.7 |
| 25 | ...ar100_retention,.1f)@\% for EuroSAT Full FT and from | `@cell(eurosat,lora_r8,3e-6,cifar100_retention,.1f)@` | 99.6 |
| 26 | ...l(eurosat,lora_r8,3e-6,cifar100_retention,.1f)@\% to | `@cell(eurosat,lora_r8,1e-4,cifar100_retention,.1f)@` | 86.4 |
| 27 | ...ains near the pretrained level throughout the grid ( | `@cell(pets,lora_r8,3e-6,cifar100_retention,.1f)@` | 100.0 |
| 28 | ...cell(pets,lora_r8,3e-6,cifar100_retention,.1f)@\% to | `@cell(pets,lora_r8,1e-4,cifar100_retention,.1f)@` | 97.7 |
| 29 | ...pets,lora_r8,1e-4,cifar100_retention,.1f)@\%; $\rho= | `@dose(pets,lora_r8,cifar100_retention)@` | -0.324 |
| 30 | ... $\rho=@dose(pets,lora_r8,cifar100_retention)@$; $p= | `@dose(pets,lora_r8,cifar100_retention,p_value,.2f)@` | 0.30 |
| 31 | ...er identical (learning rate, seed) cells, LoRA keeps | `@paired(eurosat,cifar100_retention,mean_difference,abs1)@` | 64.2 |
| 32 | ...ined CIFAR-100 accuracy than Full FT on EuroSAT ($p= | `@paired(eurosat,cifar100_retention,t_p_holm,sci)@` | 2.0\cdot 10^{-7} |
| 33 | ...comparisons; paired standardized effect size $\|d_z\|= | `@paired(eurosat,cifar100_retention,cohens_dz,abs2)@` | 4.14 |
| 34 | ...ed(eurosat,cifar100_retention,cohens_dz,abs2)@$) and | `@paired(pets,cifar100_retention,mean_difference,abs1)@` | 62.5 |
| 35 | ...tion,mean_difference,abs1)@ points more on Pets ($p= | `@paired(pets,cifar100_retention,t_p_holm,sci)@` | 2.9\cdot 10^{-5} |
| 36 | ...to the pretrained representation (mean CKA higher by | `@paired(eurosat,cka_mean,mean_difference,abs3)@` | 0.164 |
| 37 | ...@paired(eurosat,cka_mean,mean_difference,abs3)@, $p= | `@paired(eurosat,cka_mean,t_p_holm,sci)@` | 3.4\cdot 10^{-12} |
| 38 | ...t: on EuroSAT the paired difference in $\Delta H$ is | `@paired(eurosat,delta_entropy_pct,mean_difference,abs2)@` | 0.41 |
| 39 | ...lta_entropy_pct,mean_difference,abs2)@ points at $p= | `@paired(eurosat,delta_entropy_pct,t_p_value,.2f)@` | 0.49 |
| 40 | ...ta_entropy_pct,t_p_value,.2f)@$, while on Pets it is | `@paired(pets,delta_entropy_pct,mean_difference,abs2)@` | 1.63 |
| 41 | ...lta_entropy_pct,mean_difference,abs2)@ points at $p= | `@paired(pets,delta_entropy_pct,t_p_holm,sci)@` | 3.2\cdot 10^{-4} |
| 42 | ...ected operating point. On EuroSAT, Full FT selects $ | `@iso(eurosat,full_ft_lr,.0e)@` | 10^{-5} |
| 43 | ... selects $@iso(eurosat,full_ft_lr,.0e)@$ and obtains | `@iso(eurosat,full_ft_test_acc,.2f)@` | 97.19 |
| 44 | ...urosat,full_ft_test_acc,.2f)@\% target accuracy with | `@iso(eurosat,full_ft_cifar100_retention,.1f)@` | 41.2 |
| 45 | ...cifar100_retention,.1f)@\% retention; LoRA selects $ | `@iso(eurosat,lora_r8_lr,.0e)@` | 10^{-4} |
| 46 | ... selects $@iso(eurosat,lora_r8_lr,.0e)@$ and obtains | `@iso(eurosat,lora_r8_test_acc,.2f)@` | 95.16 |
| 47 | ... @iso(eurosat,lora_r8_test_acc,.2f)@\% accuracy with | `@iso(eurosat,lora_r8_cifar100_retention,.1f)@` | 86.4 |
| 48 | ...% retention. On Pets, LoRA is superior on both axes: | `@iso(pets,lora_r8_test_acc,.2f)@` | 89.88 |
| 49 | ...both axes: @iso(pets,lora_r8_test_acc,.2f)@\% versus | `@iso(pets,full_ft_test_acc,.2f)@` | 88.26 |
| 50 | ...so(pets,full_ft_test_acc,.2f)@\% target accuracy and | `@iso(pets,lora_r8_cifar100_retention,.1f)@` | 97.7 |
| 51 | ... @iso(pets,lora_r8_cifar100_retention,.1f)@\% versus | `@iso(pets,full_ft_cifar100_retention,.1f)@` | 21.8 |
| 52 | ...n set itself}, read after one epoch---reaches $\rho= | `@pred(epoch1_transfer_retention_pct,spearman_overall)@` | 0.967 |
| 53 | ...n a fair competitor. Among label-free signals, CKA ( | `@pred(cka_mean,spearman_overall)@` | 0.960 |
| 54 | ...d(cka_mean,spearman_overall)@) and embedding drift ( | `@pred(embedding_drift,spearman_overall)@` | -0.919 |
| 55 | ...verall)@) lead; a middle tier spanning $\|\rho\|$ from | `@pred(cka_last,abs_spearman_overall)@` | 0.877 |
| 56 | ...\rho\|$ from @pred(cka_last,abs_spearman_overall)@ to | `@pred(abs_delta_erf95_pct,abs_spearman_overall)@` | 0.796 |
| 57 | ...lock CKA, $\|\Delta$Gini$\|$, relative weight change ( | `@pred(weight_drift_rel,spearman_overall)@` | -0.837 |
| 58 | ...weight_drift_rel,spearman_overall)@), $\|\Delta H\|$ ( | `@pred(abs_delta_entropy_pct,spearman_overall)@` | -0.836 |
| 59 | ...all)@), $\|\Delta$ERF@0.95$\|$ and the training loss ( | `@pred(train_loss_final,spearman_overall)@` | 0.824 |
| 60 | ... computes anyway. Target-task accuracy correlates at | `@pred(test_acc,spearman_overall)@` | -0.606 |
| 61 | ...--the statistic motivating this study---reaches only | `@pred(delta_entropy_pct,spearman_overall)@` | 0.361 |
| 62 | ...evidence, uncertainty is estimated by resampling the | `@boot(cka_mean,n_units,.0f)@` | 16 |
| 63 | ... the @boot(cka_mean,n_units,.0f)@ cell means. Of the | `@stat(bootstrap_n_comparisons,.0f)@` | 10 |
| 64 | ...antage excludes zero relative to signed $\Delta H$ ( | `@boot(delta_entropy_pct,gap_vs_reference_mean,+.3f)@` | +0.601 |
| 65 | ...ference_mean,+.3f)@; 95\% confidence interval (CI) [ | `@boot(delta_entropy_pct,gap_ci_low,.3f)@` | 0.149 |
| 66 | ...rval (CI) [@boot(delta_entropy_pct,gap_ci_low,.3f)@, | `@boot(delta_entropy_pct,gap_ci_high,.3f)@` | 0.948 |
| 67 | ...ropy_pct,gap_ci_high,.3f)@]) and $\|\Delta H_{12}\|$ ( | `@boot(abs_delta_entropy_last_layer_pct,gap_vs_reference_mean,+.3f)@` | +0.226 |
| 68 | ...ntropy_last_layer_pct,gap_vs_reference_mean,+.3f)@ [ | `@boot(abs_delta_entropy_last_layer_pct,gap_ci_low,.3f)@` | 0.030 |
| 69 | ...t(abs_delta_entropy_last_layer_pct,gap_ci_low,.3f)@, | `@boot(abs_delta_entropy_last_layer_pct,gap_ci_high,.3f)@` | 0.573 |
| 70 | ...nal training loss, or weight change (for the latter: | `@boot(weight_drift_rel,gap_vs_reference_mean,+.3f)@` | +0.144 |
| 71 | ...boot(weight_drift_rel,gap_vs_reference_mean,+.3f)@ [ | `@boot(weight_drift_rel,gap_ci_low,.3f)@` | -0.018 |
| 72 | ...ean,+.3f)@ [@boot(weight_drift_rel,gap_ci_low,.3f)@, | `@boot(weight_drift_rel,gap_ci_high,.3f)@` | 0.424 |
| 73 | ...py drift becomes the strongest signal (mean $\|\rho\|= | `@pred(delta_entropy_pct,abs_spearman_within_mean)@` | 0.860 |
| 74 | ...r primary groups), ahead of relative weight change ( | `@pred(weight_drift_rel,abs_spearman_within_mean)@` | 0.823 |
| 75 | ...rel,abs_spearman_within_mean)@), log learning rate ( | `@pred(log10_lr,abs_spearman_within_mean)@` | 0.810 |
| 76 | ...n_within_mean)@), then signed $\Delta H_{12}$, CKA ( | `@pred(cka_mean,abs_spearman_within_mean)@` | 0.787 |
| 77 | ..._spearman_within_mean)@) and embedding drift between | `@pred(delta_entropy_last_layer_pct,abs_spearman_within_mean)@` | 0.788 |
| 78 | ...ntropy_last_layer_pct,abs_spearman_within_mean)@ and | `@pred(embedding_drift,abs_spearman_within_mean)@` | 0.752 |
| 79 | ...abs_spearman_within_mean)@, with $\|\Delta H\|$ last ( | `@pred(abs_delta_entropy_pct,abs_spearman_within_mean)@` | 0.656 |
| 80 | ...deployable rule must, leaves signed entropy drift at | `@pred(delta_entropy_pct,spearman_within_mean,+.3f)@` | +0.385 |
| 81 | ...an,+.3f)@ while CKA and weight change lose nothing ( | `@pred(cka_mean,spearman_within_mean,+.3f)@` | +0.787 |
| 82 | ...nothing (@pred(cka_mean,spearman_within_mean,+.3f)@, | `@pred(weight_drift_rel,spearman_within_mean,+.3f)@` | -0.823 |
| 83 | ...he best EuroSAT validation accuracy, retention spans | `@sel(eurosat,-,worst_retention)@` | 17.1 |
| 84 | ..., retention spans @sel(eurosat,-,worst_retention)@-- | `@sel(eurosat,-,oracle_retention)@` | 58.1 |
| 85 | ...ention)@--@sel(eurosat,-,oracle_retention)@\% versus | `@sel(eurosat,-,random_choice_retention)@` | 41.2 |
| 86 | ...g drift, weight change, CKA, and $\|\Delta H\|$ select | `@sel(eurosat,embedding_drift,retention)@` | 56.9 |
| 87 | ...$ select @sel(eurosat,embedding_drift,retention)@\%, | `@sel(eurosat,weight_drift_rel,retention)@` | 56.9 |
| 88 | ...on)@\%, @sel(eurosat,weight_drift_rel,retention)@\%, | `@sel(eurosat,cka_mean,retention)@` | 50.1 |
| 89 | ...ention)@\%, @sel(eurosat,cka_mean,retention)@\%, and | `@sel(eurosat,abs_delta_entropy_pct,retention)@` | 24.0 |
| 90 | ...ers only at wider margins, where the oracle rises to | `@selm(3,eurosat,-,oracle_retention)@` | 87.6 |
| 91 | ...ry EuroSAT band, while weight change reaches at most | `@selm(5,pets,weight_drift_rel,retention)@` | 54.4 |
| 92 | ...(5,pets,weight_drift_rel,retention)@\% on Pets where | `@selm(1,pets,-,oracle_retention)@` | 99.7 |
| 93 | ...pochs, CKA correlates with final retention at $\rho= | `@pred1(cka_mean,spearman_overall)@` | 0.959 |
| 94 | ...,spearman_overall)@$ and relative weight change at $ | `@pred1(weight_drift_rel,spearman_overall)@` | -0.906 |
| 95 | ...the receiver operating characteristic curve (AUC) of | `@stat(temporal_ordering.cell_level.embedding_drift.auc_oriented,.2f)@` | 1.00 |
| 96 | ...t.auc_oriented,.2f)@; relative weight change reaches | `@stat(temporal_ordering.cell_level.weight_drift_rel.auc_oriented,.2f)@` | 0.98 |
| 97 | ...auc_oriented,.2f)@, and signed entropy drift reaches | `@stat(temporal_ordering.cell_level.delta_entropy_pct.auc_oriented,.2f)@` | 0.68 |
| 98 | ... Under Full FT at $10^{-4}$, $\Delta H_{12}$ reaches | `@cell(eurosat,full_ft,1e-4,delta_entropy_last_layer_pct,+.1f)@` | -7.7 |
| 99 | ...delta_entropy_last_layer_pct,+.1f)@\% on EuroSAT and | `@cell(pets,full_ft,1e-4,delta_entropy_last_layer_pct,+.1f)@` | -18.5 |
| 100 | ...@\% on Pets, compared with block-averaged changes of | `@cell(eurosat,full_ft,1e-4,delta_entropy_pct,+.1f)@` | -2.3 |
| 101 | ...(eurosat,full_ft,1e-4,delta_entropy_pct,+.1f)@\% and | `@cell(pets,full_ft,1e-4,delta_entropy_pct,+.1f)@` | -3.2 |
| 102 | ...er. The linear probe anchors the preservation end at | `@cell(eurosat,linear_probe,1e-3,test_acc,.1f)@` | 85.1 |
| 103 | ...d at @cell(eurosat,linear_probe,1e-3,test_acc,.1f)@/ | `@cell(eurosat,linear_probe,1e-3,cifar100_retention,.1f)@` | 100.0 |
| 104 | ... are non-dominated, including Full FT at $10^{-5}$ ( | `@cell(eurosat,full_ft,1e-5,test_acc,.1f)@` | 97.2 |
| 105 | ...10^{-5}$ (@cell(eurosat,full_ft,1e-5,test_acc,.1f)@/ | `@cell(eurosat,full_ft,1e-5,cifar100_retention,.1f)@` | 41.2 |
| 106 | ...,1e-5,cifar100_retention,.1f)@), LoRA at $10^{-4}$ ( | `@cell(eurosat,lora_r8,1e-4,test_acc,.1f)@` | 95.2 |
| 107 | ...10^{-4}$ (@cell(eurosat,lora_r8,1e-4,test_acc,.1f)@/ | `@cell(eurosat,lora_r8,1e-4,cifar100_retention,.1f)@` | 86.4 |
| 108 | ...tention,.1f)@), and last-block tuning at $10^{-5}$ ( | `@cell(eurosat,last_block,1e-5,test_acc,.1f)@` | 88.6 |
| 109 | ...{-5}$ (@cell(eurosat,last_block,1e-5,test_acc,.1f)@/ | `@cell(eurosat,last_block,1e-5,cifar100_retention,.1f)@` | 97.1 |
| 110 | ...oder weights more than Full FT at $3\cdot 10^{-5}$ ( | `@cell(eurosat,last_block,1e-4,weight_drift_rel,.4f)@` | 0.0109 |
| 111 | ...urosat,last_block,1e-4,weight_drift_rel,.4f)@ versus | `@cell(eurosat,full_ft,3e-5,weight_drift_rel,.4f)@` | 0.0083 |
| 112 | ...sat,full_ft,3e-5,weight_drift_rel,.4f)@) yet retains | `@cell(eurosat,last_block,1e-4,cifar100_retention,.1f)@` | 83.8 |
| 113 | ...st_block,1e-4,cifar100_retention,.1f)@\% rather than | `@cell(eurosat,full_ft,3e-5,cifar100_retention,.1f)@` | 22.3 |
| 114 | ...jection at $10^{-4}$ similarly lowers retention from | `@cell(eurosat,lora_r8,1e-4,cifar100_retention,.1f)@` | 86.4 |
| 115 | ...l(eurosat,lora_r8,1e-4,cifar100_retention,.1f)@\% to | `@cell(eurosat,lora_r8_frozen_proj,1e-4,cifar100_retention,.1f)@` | 76.6 |
| 116 | ...,cifar100_retention,.1f)@\% and target accuracy from | `@cell(eurosat,lora_r8,1e-4,test_acc,.1f)@` | 95.2 |
| 117 | ... from @cell(eurosat,lora_r8,1e-4,test_acc,.1f)@\% to | `@cell(eurosat,lora_r8_frozen_proj,1e-4,test_acc,.1f)@` | 92.7 |
| 118 | ...sistent; within a five-point band on Pets it selects | `@selm(5,pets,weight_drift_rel,retention)@` | 54.4 |
| 119 | ...5,pets,weight_drift_rel,retention)@\% retention when | `@selm(5,pets,-,oracle_retention)@` | 99.7 |
| 120 | ...ble. For Full FT, $\alpha=0.5$ raises retention from | `@int(full_ft,1.0,transfer_retention,.1f)@` | 45.3 |
| 121 | ... from @int(full_ft,1.0,transfer_retention,.1f)@\% to | `@int(full_ft,0.5,transfer_retention,.1f)@` | 90.5 |
| 122 | ...hange; for LoRA, $\alpha=0.75$ raises retention from | `@int(lora_r8,1.0,transfer_retention,.1f)@` | 87.8 |
| 123 | ... from @int(lora_r8,1.0,transfer_retention,.1f)@\% to | `@int(lora_r8,0.75,transfer_retention,.1f)@` | 94.9 |
| 124 | ...tes strongly with eventual forgetting (mean $\|\rho\|= | `@pred(delta_entropy_pct,abs_spearman_within_mean)@` | 0.860 |
