from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class ResultStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        results: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid result JSONL at line {line_number}: {error.msg}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(f"Invalid result JSONL object at line {line_number}")
            results.append(value)
        return results

    def latest_by_key(self) -> dict[str, dict[str, Any]]:
        return {str(result["call_key"]): result for result in self.read_all()}

    def append(self, result: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
