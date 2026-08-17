#!/bin/bash
# Follow-up stages after the main campaign, in value order for the submission.
# The 3e-4 extension was dropped: LoRA already reaches a competitive Pets
# operating point at 1e-4 (90.0% vs 88.1% for Full FT), which was its purpose.
set -u
cd "$(dirname "$0")/.." || exit 1
export USE_TF=0
export TOKENIZERS_PARALLELISM=false
LOG=study2/campaign2.log

until grep -q "campaign complete" study2/campaign.log 2>/dev/null; do sleep 60; done
echo "=== $(date '+%H:%M:%S') main campaign finished, starting follow-up ===" >>"$LOG"

# Does the predictor ranking survive a different backbone? (6 runs)
echo "=== $(date '+%H:%M:%S') STAGE 6 ViT-B/16 generality ===" >>"$LOG"
python3 -u -m study2.run_study2 --datasets eurosat --methods full_ft lora_r8 \
    --learning-rates 1e-5 3e-5 1e-4 --seeds 7 --epochs 6 \
    --model openai/clip-vit-base-patch16 >>"$LOG" 2>&1

# Can the predicted trade-off be acted on? (2 sweeps)
echo "=== $(date '+%H:%M:%S') STAGE 7 intervention ===" >>"$LOG"
python3 -u -m study2.run_intervention --dataset eurosat --method full_ft --lr 3e-5 \
    --seed 7 --alphas 0.0 0.25 0.5 0.75 1.0 >>"$LOG" 2>&1
python3 -u -m study2.run_intervention --dataset eurosat --method lora_r8 --lr 1e-4 \
    --seed 7 --alphas 0.0 0.25 0.5 0.75 1.0 >>"$LOG" 2>&1

echo "=== $(date '+%H:%M:%S') follow-up complete ===" >>"$LOG"
