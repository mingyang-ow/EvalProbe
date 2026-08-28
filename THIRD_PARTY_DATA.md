# Third-party data

EvalProbe uses [RAGTruth](https://github.com/ParticleMedia/RAGTruth) as its benchmark. The QA portion identifies its upstream source as `MARCO` and contains MS MARCO question and passage material.

This repository intentionally does not redistribute RAGTruth JSONL files, MS MARCO passages, or processed artifacts containing substantial third-party text. The official RAGTruth repository includes an MIT `LICENSE`; that does not by itself establish new redistribution rights for all upstream source material. MS MARCO publishes separate [terms and conditions](https://microsoft.github.io/msmarco/). Users are responsible for reviewing the current source terms that apply to their use.

Obtain the two benchmark files from the official source:

```bash
uv run python scripts/fetch_ragtruth.py
```

This downloads `dataset/source_info.jsonl` and `dataset/response.jsonl` from the official RAGTruth GitHub repository into the gitignored `data/raw/` directory. The Phase 0 summary records SHA-256 fingerprints of the files actually audited.

Tracked outputs are limited to code, configuration, identifiers, numeric derived features, aggregated statistics, and reports that do not reproduce source passages. The text-bearing manual conversion audit is generated locally and gitignored.
