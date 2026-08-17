#!/bin/bash
# Follow-up stages. Waits for the first campaign to finish, then runs the
# generality and intervention checks. Resumable like the first campaign.
set -u
cd "$(dirname "$0")/.." || exit 1
export USE_TF=0
export TOKENIZERS_PARALLELISM=false
LOG=study2/campaign2.log

until grep -q "campaign complete" study2/campaign.log 2>/dev/null; do sleep 60; done
echo "=== $(date '+%H:%M:%S') first campaign finished, starting follow-up ===" >>"$LOG"

# Stage 5: can the predicted trade-off be acted on? Encoder interpolation.
echo "=== $(date '+%H:%M:%S') STAGE 5 intervention ===" >>"$LOG"
python3 -u -m study2.run_intervention --dataset eurosat --method full_ft --lr 3e-5 \
    --seed 7 --alphas 0.0 0.25 0.5 0.75 1.0 >>"$LOG" 2>&1
python3 -u -m study2.run_intervention --dataset eurosat --method lora_r8 --lr 1e-4 \
    --seed 7 --alphas 0.0 0.25 0.5 0.75 1.0 >>"$LOG" 2>&1

# Stage 6: does the predictor ranking survive a different backbone?
echo "=== $(date '+%H:%M:%S') STAGE 6 ViT-B/16 spot check ===" >>"$LOG"
python3 -u -m study2.run_study2 --datasets eurosat --methods full_ft lora_r8 \
    --learning-rates 1e-5 1e-4 --seeds 7 --epochs 6 \
    --model openai/clip-vit-base-patch16 >>"$LOG" 2>&1

echo "=== $(date '+%H:%M:%S') follow-up complete ===" >>"$LOG"
