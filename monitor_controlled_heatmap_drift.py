#!/usr/bin/env python3
"""Monitor the controlled heatmap-drift matrix and trigger final analysis."""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
METRICS_DIR = PROJECT_DIR / "outputs" / "metrics"
LOG_DIR = PROJECT_DIR / "outputs" / "logs"
EXPECTED_HISTORY_COUNT = 80
POLL_SECONDS = 60


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def count_histories() -> int:
    return sum(1 for _ in METRICS_DIR.glob("CHD*_history.json"))


def matrix_process_running() -> bool:
    result = subprocess.run(
        ["pgrep", "-f", "python run_controlled_heatmap_drift.py"],
        cwd=PROJECT_DIR,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def run_final_analysis() -> int:
    cmd = [sys.executable, "analyze_controlled_heatmap_drift.py", "--refresh-zero-shot"]
    completed = subprocess.run(cmd, cwd=PROJECT_DIR, check=False)
    return completed.returncode


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[{utc_now()}] Monitor started", flush=True)

    while True:
        history_count = count_histories()
        running = matrix_process_running()
        print(
            f"[{utc_now()}] histories={history_count}/{EXPECTED_HISTORY_COUNT} matrix_running={running}",
            flush=True,
        )

        if history_count >= EXPECTED_HISTORY_COUNT:
            print(f"[{utc_now()}] Matrix complete; refreshing zero-shot summaries", flush=True)
            rc = run_final_analysis()
            print(f"[{utc_now()}] Final analysis exit_code={rc}", flush=True)
            return rc

        if not running:
            print(
                f"[{utc_now()}] Matrix process not running before completion; leaving without analysis",
                flush=True,
            )
            return 1

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
