from pathlib import Path

import pytest
import yaml

from evalprobe.cli import _load_config


def test_frozen_config_enforces_reference_and_sampling_rules() -> None:
    config = _load_config(Path("configs/phase0.yaml"))
    assert config["reference"]["implicit_true_is_unsupported"] is True
    assert config["unsupported_strata"]["method"] == "train_tertiles"
    assert config["sampling"]["max_per_source_id"] == 1


def test_inconsistent_frozen_totals_fail_loudly(tmp_path: Path) -> None:
    config = _load_config(Path("configs/phase0.yaml"))
    config["sampling"]["total"] = 59
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="sampling.total"):
        _load_config(path)
