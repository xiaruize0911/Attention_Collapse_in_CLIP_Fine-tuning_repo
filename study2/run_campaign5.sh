#!/bin/bash
# Reviewer condition #4: the ViT-B/16 arm was single-seed (n=3 per group), so a
# perfect |rho| there arises from 1 in 3 permutations under the null. Two more
# seeds per cell make the polarity claim testable instead of anecdotal.
set -u
cd "$(dirname "$0")/.." || exit 1
export USE_TF=0
export TOKENIZERS_PARALLELISM=false
LOG=study2/campaign2.log
echo "=== $(date '+%H:%M:%S') STAGE 8 ViT-B/16 seeds 11 and 19 ===" >>"$LOG"
python3 -u -m study2.run_study2 --datasets eurosat --methods full_ft lora_r8 \
    --learning-rates 1e-5 3e-5 1e-4 --seeds 11 19 --epochs 6 \
    --model openai/clip-vit-base-patch16 >>"$LOG" 2>&1
echo "=== $(date '+%H:%M:%S') STAGE 8 complete ===" >>"$LOG"
