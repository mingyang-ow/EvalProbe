from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from evalprobe.phase0.audit import build_pilot, run_audit


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Phase 0 config must be a YAML mapping")
    if value.get("reference", {}).get("implicit_true_is_unsupported") is not True:
        raise ValueError("Phase 0 requires implicit_true annotations to remain unsupported")
    if value.get("unsupported_strata", {}).get("method") != "train_tertiles":
        raise ValueError("Phase 0 requires train-derived burden tertiles")
    if value.get("sampling", {}).get("max_per_source_id") != 1:
        raise ValueError("Phase 0 requires maximum one response per source_id")
    sampling = value.get("sampling", {})
    strata = value.get("unsupported_strata", {})
    if sampling.get("total") != sampling.get("supported", 0) + sampling.get("unsupported", 0):
        raise ValueError("sampling.total must equal supported plus unsupported")
    if sampling.get("unsupported") != sum(strata.get(key, 0) for key in ("low", "medium", "high")):
        raise ValueError("unsupported stratum quotas must sum to sampling.unsupported")
    return value


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="evalprobe")
    commands = root.add_subparsers(dest="command", required=True)
    phase0 = commands.add_parser("phase0", help="RAGTruth Phase 0 operations")
    phase0_commands = phase0.add_subparsers(dest="phase0_command", required=True)
    for name in ("audit", "build-pilot"):
        command = phase0_commands.add_parser(name)
        command.add_argument("--config", type=Path, default=Path("configs/phase0.yaml"))
        command.add_argument("--data-dir", type=Path, default=Path("data/raw"))
        command.add_argument("--output-dir", type=Path, default=Path("reports/phase0"))
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = _load_config(args.config)
    if args.phase0_command == "audit":
        summary = run_audit(args.data_dir, args.output_dir, config)
        print(
            f"Audited {summary['dataset']['qa_response_count']} QA responses; "
            f"blockers={len(summary['integrity']['blockers'])}."
        )
        print(f"Report: {args.output_dir / 'report.md'}")
        return 0
    pilot = build_pilot(args.output_dir, config)
    print(
        f"Built {pilot['selected_total']}-response pilot with "
        f"{pilot['unique_source_ids']} unique source IDs."
    )
    print(f"Manifest: {args.output_dir / 'pilot_manifest.jsonl'}")
    return 0
