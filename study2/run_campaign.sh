#!/bin/bash
# Sequential campaign for the protocol-hardened study. Every stage is resumable:
# a run whose JSON already exists in study2/results is skipped, so the script can
# be interrupted and restarted at any time.
set -u
cd "$(dirname "$0")/.." || exit 1
export USE_TF=0
export TOKENIZERS_PARALLELISM=false
LOG=study2/campaign.log
EP=6

stage() { echo "=== $(date '+%H:%M:%S') STAGE $* ===" >>"$LOG"; }

# Stage 1: matched-learning-rate core grid, EuroSAT (24 runs)
stage "1 eurosat core"
python3 -u -m study2.run_study2 --datasets eurosat --methods full_ft lora_r8 \
    --learning-rates 3e-6 1e-5 3e-5 1e-4 --seeds 7 11 19 --epochs $EP >>"$LOG" 2>&1

# Stage 2: matched-learning-rate core grid, Oxford-IIIT Pets (24 runs)
stage "2 pets core"
python3 -u -m study2.run_study2 --datasets pets --methods full_ft lora_r8 \
    --learning-rates 3e-6 1e-5 3e-5 1e-4 --seeds 7 11 19 --epochs $EP >>"$LOG" 2>&1

# Stage 3: reference points that bracket the trade-off (12 runs)
stage "3 reference points"
python3 -u -m study2.run_study2 --datasets eurosat pets --methods linear_probe \
    --learning-rates 1e-3 --seeds 7 11 19 --epochs $EP >>"$LOG" 2>&1
python3 -u -m study2.run_study2 --datasets eurosat --methods last_block \
    --learning-rates 1e-5 1e-4 --seeds 7 11 19 --epochs $EP >>"$LOG" 2>&1

# Stage 4: does the trained visual projection explain LoRA's transfer loss? (6 runs)
stage "4 frozen-projection ablation"
python3 -u -m study2.run_study2 --datasets eurosat --methods lora_r8_frozen_proj \
    --learning-rates 1e-5 1e-4 --seeds 7 11 19 --epochs $EP >>"$LOG" 2>&1

stage "campaign complete"
