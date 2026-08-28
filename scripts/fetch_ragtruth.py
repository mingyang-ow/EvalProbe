from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset"
FILES = ("source_info.jsonl", "response.jsonl")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download official RAGTruth JSONL files")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        destination = args.output_dir / filename
        print(f"Downloading {filename} to {destination}")
        urllib.request.urlretrieve(f"{BASE_URL}/{filename}", destination)  # noqa: S310
    print("Raw files are gitignored. Review THIRD_PARTY_DATA.md before use.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
