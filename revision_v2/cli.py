from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import analyze
from .config import RevisionConfig, load_config
from .training import prepare_data, run_appendix, run_backbone_confirmation, run_lr_sweep, run_main, run_pilot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Revision v2 CLIP structural preservation pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML or JSON config file.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in [
        "prepare-data",
        "run-pilot",
        "run-main",
        "run-lr-sweep",
        "run-backbone-confirmation",
        "run-appendix",
        "analyze",
        "build-paper-assets",
    ]:
        subparsers.add_parser(command)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config) if args.config else RevisionConfig()

    if args.command == "prepare-data":
        payload = prepare_data(config)
    elif args.command == "run-pilot":
        payload = [result.summary_path.as_posix() for result in run_pilot(config)]
    elif args.command == "run-main":
        payload = [result.summary_path.as_posix() for result in run_main(config)]
    elif args.command == "run-lr-sweep":
        payload = [result.summary_path.as_posix() for result in run_lr_sweep(config)]
    elif args.command == "run-backbone-confirmation":
        payload = [result.summary_path.as_posix() for result in run_backbone_confirmation(config)]
    elif args.command == "run-appendix":
        payload = [result.summary_path.as_posix() for result in run_appendix(config)]
    elif args.command in {"analyze", "build-paper-assets"}:
        payload = analyze(config)
    else:
        raise ValueError(f"Unhandled command: {args.command}")

    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
