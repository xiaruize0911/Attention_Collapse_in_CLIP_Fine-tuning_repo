#!/bin/bash
# Follow-up stages, in priority order. Waits for the main campaign to finish.
set -u
cd "$(dirname "$0")/.." || exit 1
export USE_TF=0
export TOKENIZERS_PARALLELISM=false
LOG=study2/campaign2.log

until grep -q "campaign complete" study2/campaign.log 2>/dev/null; do sleep 60; done
echo "=== $(date '+%H:%M:%S') main campaign finished, starting follow-up ===" >>"$LOG"

# Stage 5: extend the matched grid to a fifth, more aggressive learning rate, so
# that LoRA reaches competitive target accuracy on Pets and Full FT is clearly
# past the point of damage on both datasets.
echo "=== $(date '+%H:%M:%S') STAGE 5 lr 3e-4 extension ===" >>"$LOG"
python3 -u -m study2.run_study2 --datasets eurosat pets --methods full_ft lora_r8 \
    --learning-rates 3e-4 --seeds 7 11 19 --epochs 6 >>"$LOG" 2>&1

# Stage 6: can the predicted trade-off be acted on? Encoder interpolation.
echo "=== $(date '+%H:%M:%S') STAGE 6 intervention ===" >>"$LOG"
python3 -u -m study2.run_intervention --dataset eurosat --method full_ft --lr 3e-5 \
    --seed 7 --alphas 0.0 0.25 0.5 0.75 1.0 >>"$LOG" 2>&1
python3 -u -m study2.run_intervention --dataset eurosat --method lora_r8 --lr 1e-4 \
    --seed 7 --alphas 0.0 0.25 0.5 0.75 1.0 >>"$LOG" 2>&1

# Stage 7: does the predictor ranking survive a different backbone?
echo "=== $(date '+%H:%M:%S') STAGE 7 ViT-B/16 spot check ===" >>"$LOG"
python3 -u -m study2.run_study2 --datasets eurosat --methods full_ft lora_r8 \
    --learning-rates 1e-5 1e-4 --seeds 7 --epochs 6 \
    --model openai/clip-vit-base-patch16 >>"$LOG" 2>&1

echo "=== $(date '+%H:%M:%S') follow-up complete ===" >>"$LOG"
