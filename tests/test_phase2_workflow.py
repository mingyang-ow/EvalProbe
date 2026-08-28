import json
from pathlib import Path

from evalprobe.phase2.workflow import (
    load_phase2_config,
    sha256_file,
    validate_frozen_manifest,
    validate_train_freeze,
)

REPOSITORY_ROOT = Path(__file__).parents[1]


def test_frozen_test_config_and_manifest_identity_are_exact() -> None:
    config = load_phase2_config(REPOSITORY_ROOT / "configs/phase2.yaml")
    manifest_path = REPOSITORY_ROOT / config["frozen_pilot"]["path"]
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]

    assert sha256_file(manifest_path) == config["frozen_pilot"]["sha256"]
    assert validate_frozen_manifest(rows, config) == {
        "record_count": 60,
        "reference_counts": {"SUPPORTED": 30, "UNSUPPORTED": 30},
        "unsupported_stratum_counts": {"high": 10, "low": 10, "medium": 10},
        "unique_record_ids": 60,
        "unique_source_ids": 60,
    }


def test_train_freeze_has_no_unresolved_methodology_defect() -> None:
    gate = validate_train_freeze(REPOSITORY_ROOT / "reports/review/review_summary.json")

    assert gate["sentence_v2_pass"] == 20
    assert gate["phase1c_reviewed"] == 7
    assert gate["phase1c_unreviewed"] == 0
    assert gate["phase1c_classification_counts"] == {
        "BENCHMARK_AMBIGUITY": 7,
        "JUDGE_ERROR": 0,
        "REFERENCE_MAPPING_ARTIFACT": 0,
        "RUBRIC_AMBIGUITY": 0,
        "SEGMENTATION_DEFECT": 0,
    }
