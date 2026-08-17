# Fresh local rerun verification

Date: August 13, 2026  
Device: Apple MPS  
Dataset: EuroSAT  
Training subset: 640 randomly selected images  
Evaluation: full 2,700-image provided test split plus the fixed 200-image attention subset  
Protocol: one epoch, seed 42, learning rate `1e-5`

These short runs verify the complete training, evaluation, attention-metric, and checkpoint paths. They are deliberately identified with `_e1_n640` and are not used as evidence in the paper.

| Method | Train accuracy | Evaluation accuracy | Baseline entropy | Final entropy | Entropy change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full FT | 44.84% | 63.93% | 3.68438 | 3.70878 | +0.6621% |
| LoRA r=8 | 14.84% | 17.56% | 3.68438 | 3.68432 | -0.0018% |

Raw histories:

- `outputs/metrics/CHD_fullft_eurosat_lr1e-5_seed42_e1_n640_history.json`
- `outputs/metrics/CHD_lora_r8_eurosat_lr1e-5_seed42_e1_n640_history.json`

The difference is expected for this intentionally tiny run: at the same small learning rate, Full FT moves many more parameters, while LoRA needs a larger rate or more epochs. That behavior is consistent with the underfitting interaction highlighted in the controlled paper.
