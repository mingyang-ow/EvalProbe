from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml

from evalprobe.phase0.audit import build_pilot, run_audit
from evalprobe.phase1.analysis import analyze_canary
from evalprobe.phase1.persistence import ResultStore
from evalprobe.phase1.provider import OpenAIResponsesJudge
from evalprobe.phase1.runner import (
    CanaryExecutionError,
    build_plan,
    execute_plan,
    manifest_rows,
    preflight_summary,
    write_dry_run,
)
from evalprobe.phase1c.analysis import analyze_phase1c
from evalprobe.phase1c.diagnostics import (
    corpus_segmentation_diagnostics,
    regenerate_phase0_v2,
)
from evalprobe.phase1c.workflow import (
    build_phase1c_plan,
    load_phase1c_config,
    reusable_whole_results,
)
from evalprobe.phase2.analysis import analyze_frozen_test
from evalprobe.phase2.workflow import (
    build_frozen_test_plan,
    load_phase2_config,
    phase2_preflight_summary,
    write_phase2_dry_run,
)
from evalprobe.phase3.review_set import (
    load_phase3_review_items,
    phase3_adjudication_summary,
    prepare_phase3_review_set,
)
from evalprobe.review.diagnostics import aggregate_suspicious_units
from evalprobe.review.loaders import (
    load_phase1_canary_records,
    phase1_current_local_disagreement_targets,
)
from evalprobe.review.storage import load_adjudications, review_summary, write_safe_json


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


def _load_phase1_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Phase 1 config must be a YAML mapping")
    canary = value.get("canary", {})
    judge = value.get("judge", {})
    budget = value.get("budget", {})
    if canary.get("split") != "train":
        raise ValueError("Phase 1A canary must use TRAIN only")
    if canary.get("total") != 6 or canary.get("supported") != 3 or canary.get("unsupported") != 3:
        raise ValueError("Phase 1A requires a 3/3 six-record canary")
    if sum(canary.get("unsupported_strata", {}).values()) != canary.get("unsupported"):
        raise ValueError("Canary unsupported strata must sum to three")
    if canary.get("max_per_source_id") != 1 or canary.get("expected_calls") != 12:
        raise ValueError("Phase 1A requires 12 calls and one record per source")
    if judge.get("model") != "gpt-5.6-sol" or judge.get("reasoning_effort") != "low":
        raise ValueError("Phase 1A requires gpt-5.6-sol with low reasoning")
    if judge.get("automatic_retries") != 0 or judge.get("fallback_models") != 0:
        raise ValueError("Phase 1A prohibits retries and fallback models")
    if budget.get("maximum_paid_calls") != 12 or float(budget.get("hard_cap_usd", 0)) != 0.50:
        raise ValueError("Phase 1A requires a 12-call, USD $0.50 hard cap")
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
    phase1 = commands.add_parser("phase1", help="Judge contract and TRAIN canary")
    phase1_commands = phase1.add_subparsers(dest="phase1_command", required=True)
    canary = phase1_commands.add_parser("canary")
    mode = canary.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    canary.add_argument("--max-cost-usd", type=float)
    canary.add_argument("--config", type=Path, default=Path("configs/phase1.yaml"))
    canary.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    canary.add_argument(
        "--features",
        type=Path,
        default=Path("reports/phase0/derived_features.jsonl"),
    )
    canary.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/phase1/train-canary-v1"),
    )
    review = commands.add_parser("review", help="Local human-adjudication operations")
    review_commands = review.add_subparsers(dest="review_command", required=True)
    summary = review_commands.add_parser("summary", help="Summarize safe human decisions")
    summary.add_argument(
        "--adjudications",
        type=Path,
        default=Path("reports/review/adjudications.jsonl"),
    )
    summary.add_argument(
        "--output",
        type=Path,
        default=Path("reports/review/review_summary.json"),
    )
    diagnostics = review_commands.add_parser(
        "diagnostics", help="Count suspicious deterministic sentence units"
    )
    diagnostics.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    diagnostics.add_argument(
        "--output",
        type=Path,
        default=Path("reports/review/suspicious_units_summary.json"),
    )
    phase1c = commands.add_parser("phase1c", help="Deterministic segmentation repair rerun")
    phase1c_commands = phase1c.add_subparsers(dest="phase1c_command", required=True)
    repaired_diagnostics = phase1c_commands.add_parser("diagnostics")
    repaired_diagnostics.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    repaired_diagnostics.add_argument(
        "--manual-v1", type=Path, default=Path("reports/phase0/manual_audit.jsonl")
    )
    repaired_diagnostics.add_argument(
        "--manual-v2", type=Path, default=Path("reports/phase0/manual_audit_v2.jsonl")
    )
    repaired_diagnostics.add_argument(
        "--adjudications",
        type=Path,
        default=Path("reports/review/adjudications.jsonl"),
    )
    repaired_diagnostics.add_argument("--output-dir", type=Path, default=Path("reports/phase1c"))
    repaired_canary = phase1c_commands.add_parser("canary")
    repaired_mode = repaired_canary.add_mutually_exclusive_group(required=True)
    repaired_mode.add_argument("--dry-run", action="store_true")
    repaired_mode.add_argument("--execute", action="store_true")
    repaired_canary.add_argument("--max-cost-usd", type=float)
    repaired_canary.add_argument("--config", type=Path, default=Path("configs/phase1c.yaml"))
    repaired_canary.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    repaired_canary.add_argument(
        "--features", type=Path, default=Path("reports/phase0/derived_features.jsonl")
    )
    repaired_canary.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/phase1c/train-canary-segmentation-v2"),
    )
    phase2 = commands.add_parser("phase2", help="Frozen TEST experiment")
    phase2_mode = phase2.add_mutually_exclusive_group(required=True)
    phase2_mode.add_argument("--dry-run", action="store_true")
    phase2_mode.add_argument("--execute", action="store_true")
    phase2.add_argument("--max-cost-usd", type=float)
    phase2.add_argument("--config", type=Path, default=Path("configs/phase2.yaml"))
    phase2.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    phase2.add_argument("--output-dir", type=Path, default=Path("reports/phase2/frozen-test-v1"))
    phase3 = commands.add_parser("phase3", help="Frozen TEST human error analysis")
    phase3.add_argument("--prepare", action="store_true", required=True)
    phase3.add_argument(
        "--manifest",
        type=Path,
        default=Path("reports/phase2/frozen-test-v1/manifest.jsonl"),
    )
    phase3.add_argument(
        "--results",
        type=Path,
        default=Path("reports/phase2/frozen-test-v1/results.jsonl"),
    )
    phase3.add_argument(
        "--queue",
        type=Path,
        default=Path("reports/phase2/frozen-test-v1/review_queue.jsonl"),
    )
    phase3.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/phase3/frozen-test-error-analysis"),
    )
    phase3.add_argument("--seed", type=int, default=20260828)
    phase3.add_argument("--judge-only-target", type=int, default=20)
    phase3.add_argument("--max-per-record", type=int, default=2)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "phase1":
        return _run_phase1(args)
    if args.command == "review":
        return _run_review(args)
    if args.command == "phase1c":
        return _run_phase1c(args)
    if args.command == "phase2":
        return _run_phase2(args)
    if args.command == "phase3":
        return _run_phase3(args)
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


def _print_preflight(summary: dict[str, Any]) -> None:
    print(f"Model: {summary['model']}")
    print(f"Reasoning effort: {summary['reasoning_effort']}")
    print(
        f"Selected {str(summary['selected_split']).upper()} records: "
        f"{summary['selected_record_count']}"
    )
    print(
        f"Expected calls: {summary['expected_call_count']} "
        f"({summary['whole_call_count']} whole + {summary['local_call_count']} local)"
    )
    print(
        f"Approximate input: {summary['approximate_input_characters']} characters / "
        f"{summary['approximate_input_tokens']} tokens"
    )
    print(f"Configured max output: {summary['max_output_tokens']}")
    print(f"Conservative estimated maximum cost: ${summary['estimated_max_cost_usd']:.6f}")
    print(f"Hard spend limit: ${summary['hard_spend_limit_usd']:.2f}")


def _run_phase1(args: argparse.Namespace) -> int:
    config = _load_phase1_config(args.config)
    plan = build_plan(config, args.data_dir, args.features)
    configured_cap = float(config["budget"]["hard_cap_usd"])
    max_cost = configured_cap if args.max_cost_usd is None else args.max_cost_usd
    summary = preflight_summary(plan, max_cost)
    _print_preflight(summary)
    if args.dry_run:
        write_dry_run(plan, args.output_dir, max_cost)
        print("Dry run complete: 0 network calls.")
        print(f"Summary: {args.output_dir / 'dry_run.json'}")
        return 0
    if args.max_cost_usd is None:
        raise SystemExit("--execute requires explicit --max-cost-usd")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required for --execute")
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=0)
    judge = OpenAIResponsesJudge(client)
    try:
        results = execute_plan(plan, judge, args.output_dir, max_cost)
    except CanaryExecutionError as error:
        raise SystemExit(str(error)) from error
    analysis = analyze_canary(manifest_rows(plan), results, args.output_dir, max_cost)
    print(f"Completed calls: {analysis['completed_calls']} / {analysis['expected_calls']}")
    print(f"Estimated API cost: ${analysis['total_estimated_cost_usd']:.6f}")
    print(f"Report: {args.output_dir / 'canary_report.md'}")
    return 0


def _run_review(args: argparse.Namespace) -> int:
    if args.review_command == "summary":
        repository_root = Path.cwd()
        phase1_dir = repository_root / "reports/phase1/train-canary-v1"
        phase1c_dir = repository_root / "reports/phase1c/train-canary-segmentation-v2"
        phase1c_records = load_phase1_canary_records(
            phase1c_dir / "manifest.jsonl",
            phase1c_dir / "results.jsonl",
            repository_root / "reports/phase0/derived_features.jsonl",
            repository_root / "data/raw",
            whole_results_path=phase1_dir / "results.jsonl",
            run_id_override="train-canary-segmentation-v2",
            local_units_version="sentence-v2",
        )
        phase1c_targets = phase1_current_local_disagreement_targets(phase1c_records)
        summary = review_summary(
            load_adjudications(args.adjudications), phase1c_targets=phase1c_targets
        )
        phase3_path = repository_root / (
            "reports/phase3/frozen-test-error-analysis/review_set.jsonl"
        )
        decisions = load_adjudications(args.adjudications)
        if phase3_path.is_file():
            summary["phase3_test_error_analysis"] = phase3_adjudication_summary(
                load_phase3_review_items(phase3_path), decisions
            )
        write_safe_json(args.output, summary)
        phase0_versions = summary["phase0_sentence_audit_versions"]
        phase1 = summary["phase1a_disagreements"]
        phase1c = summary["phase1c_sentence_v2_disagreements"]
        for version, phase0 in phase0_versions.items():
            print(f"Phase 0 sentence audit ({version})")
            for status, count in phase0["status_counts"].items():
                print(f"{status}: {count}")
        print("Phase 1A historical disagreements")
        for classification, count in phase1["classification_counts"].items():
            print(f"{classification}: {count}")
        print("Phase 1C sentence-v2 current disagreements")
        for classification, count in phase1c["classification_counts"].items():
            print(f"{classification}: {count}")
        print(f"Unreviewed: {phase1c['unreviewed_count']}")
        if "phase3_test_error_analysis" in summary:
            print("Phase 3 frozen TEST error analysis")
            for group, counts in summary["phase3_test_error_analysis"]["groups"].items():
                print(f"{group}: {counts['reviewed_count']} reviewed / {counts['target_count']}")
        print(f"Summary: {args.output}")
        return 0
    diagnostics = aggregate_suspicious_units(args.data_dir)
    write_safe_json(args.output, diagnostics)
    for split, counts in diagnostics["splits"].items():
        print(
            f"{split.upper()}: {counts['suspicious_unit_count']} suspicious / "
            f"{counts['sentence_unit_count']} sentence units"
        )
    print(f"Diagnostics: {args.output}")
    return 0


def _run_phase1c(args: argparse.Namespace) -> int:
    repository_root = Path.cwd()
    if args.phase1c_command == "diagnostics":
        phase0 = regenerate_phase0_v2(
            args.manual_v1,
            args.manual_v2,
            args.adjudications,
            args.output_dir / "phase0_v2_summary.json",
        )
        corpus = corpus_segmentation_diagnostics(
            args.data_dir, args.output_dir / "corpus_segmentation_diagnostics.json"
        )
        spans = corpus["span_integrity"]
        print(
            f"Phase 0 v2: {phase0['record_count']} records; "
            f"changed prior failures={phase0['previous_failures_changed_count']}."
        )
        print(
            f"Corpus list markers: {corpus['overall']['list_marker_only_before']} -> "
            f"{corpus['overall']['list_marker_only_after']}."
        )
        print(
            f"Span validation: {spans['matched']} matched; mismatch={spans['mismatch']}; "
            f"malformed={spans['malformed']}; out_of_range={spans['out_of_range']}."
        )
        return 0

    config = load_phase1c_config(args.config)
    repaired_plan, base_plan = build_phase1c_plan(
        config, args.data_dir, args.features, repository_root
    )
    reuse_path = repository_root / str(config["reuse"]["phase1a_results"])
    reusable = reusable_whole_results(base_plan, reuse_path)
    configured_cap = float(config["budget"]["hard_cap_usd"])
    max_cost = configured_cap if args.max_cost_usd is None else args.max_cost_usd
    summary = preflight_summary(repaired_plan, max_cost)
    _print_preflight(summary)
    print(f"Reusable Phase 1A whole predictions: {len(reusable)}")
    if args.dry_run:
        write_dry_run(repaired_plan, args.output_dir, max_cost)
        print("Dry run complete: 0 network calls; six local calls planned.")
        return 0
    if args.max_cost_usd is None:
        raise SystemExit("--execute requires explicit --max-cost-usd")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required for --execute")
    from openai import OpenAI

    judge = OpenAIResponsesJudge(OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=0))
    try:
        results = execute_plan(repaired_plan, judge, args.output_dir, max_cost)
    except CanaryExecutionError as error:
        raise SystemExit(str(error)) from error
    analysis = analyze_phase1c(
        repaired_plan,
        base_plan,
        repository_root / str(config["reuse"]["phase1a_manifest"]),
        reuse_path,
        results,
        reusable,
        args.output_dir,
        max_cost,
    )
    print(
        f"Completed local calls: {analysis['local_calls']['completed']} / "
        f"{analysis['local_calls']['planned']}"
    )
    print(f"Estimated API cost: ${analysis['total_estimated_cost_usd']:.6f}")
    print(f"Comparison: {args.output_dir / 'comparison.md'}")
    return 0


def _run_phase2(args: argparse.Namespace) -> int:
    repository_root = Path.cwd()
    config = load_phase2_config(args.config)
    plan, freeze = build_frozen_test_plan(config, args.data_dir, repository_root)
    configured_cap = float(config["budget"]["hard_cap_usd"])
    max_cost = configured_cap if args.max_cost_usd is None else args.max_cost_usd
    summary = phase2_preflight_summary(plan, freeze, args.output_dir, max_cost, repository_root)
    _print_preflight(summary)
    print(f"Whole prompt: {summary['prompt_versions']['whole']}")
    print(f"Local prompt: {summary['prompt_versions']['local']}")
    print(f"Local units: {summary['local_units_version']}")
    print(f"Frozen manifest SHA-256: {summary['frozen_manifest_sha256']}")
    print(f"Pre-execution gate: {summary['pre_execution_gate'].upper()}")
    if args.dry_run:
        summary = write_phase2_dry_run(plan, freeze, args.output_dir, max_cost, repository_root)
        print("Dry run complete: 0 network calls; 120 TEST calls planned.")
        if summary["pre_execution_gate"] != "pass":
            print("BLOCKED BEFORE TEST: pre-execution gate failed.")
        print(f"Summary: {args.output_dir / 'dry_run.json'}")
        return 0
    if args.max_cost_usd is None:
        raise SystemExit("--execute requires explicit --max-cost-usd")
    if summary["pre_execution_gate"] != "pass":
        raise SystemExit("BLOCKED BEFORE TEST: pre-execution gate failed; no calls made")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required for --execute")
    from openai import OpenAI

    judge = OpenAIResponsesJudge(OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=0))
    try:
        results = execute_plan(plan, judge, args.output_dir, max_cost)
    except CanaryExecutionError as error:
        partial = ResultStore(args.output_dir / "results.jsonl").read_all()
        if partial:
            analyze_frozen_test(plan, partial, args.output_dir)
        raise SystemExit(str(error)) from error
    analysis = analyze_frozen_test(plan, results, args.output_dir)
    completed = analysis["operational"]["completed_calls"]
    print(f"Completed TEST calls: {completed} / {len(plan.calls)}")
    print(f"Operational failures: {analysis['operational']['operational_failure_count']}")
    print(f"Estimated API cost: ${analysis['operational']['cost_usd']:.6f}")
    print(f"Report: {args.output_dir / 'report.md'}")
    return 0


def _run_phase3(args: argparse.Namespace) -> int:
    summary = prepare_phase3_review_set(
        args.manifest,
        args.results,
        args.queue,
        args.output_dir,
        seed=args.seed,
        judge_only_target=args.judge_only_target,
        max_per_record=args.max_per_record,
    )
    groups = summary["group_counts"]
    print(f"Whole disagreements: {groups['WHOLE_DISAGREEMENTS']}")
    print(f"Local REFERENCE_ONLY: {groups['LOCAL_REFERENCE_ONLY']}")
    print(f"Sampled local JUDGE_ONLY: {groups['LOCAL_JUDGE_ONLY_SAMPLE']}")
    print(f"Total human review items: {summary['total_review_items']}")
    print("Provider calls: 0; API cost: $0.00")
    print(f"Review set: {args.output_dir / 'review_set.jsonl'}")
    return 0
