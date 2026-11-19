# Dataset Card: arm-a-forget-train

## Summary

- **Run timestamp**: 2026-11-13T00:09:05.118368Z
- **Wall clock time**: 1.6s
- **Pipeline config hash**: `9385a3a798be0904`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 1503 | 1503 | — | 0 | 1503 |
| SchemaGate | 1503 | 1503 | 0 | 0 | 1503 |
| ExactDeduplicator | 1503 | 1246 | — | — | 1246 |
| TextCleaner | 1246 | 1246 | — | — | 1246 |
| CorpusExporter | — | 1246 | — | — | 1246 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 1503/0 passed (N/A), 0 rejected
- **SchemaGate**: 1503/1503 passed (100.0%), 0 rejected


**Rejection reasons:**
  - None

## Deduplication

  - exact_duplicates_removed: 257

## Known Limitations

- No MinHash deduplication applied.
- Table extraction requires `extract_tables: true` on the PDF reader and MinerU with GPU support. Disabled by default.
- Token counting uses a whitespace tokenizer by default. For exact LLM context window
  enforcement, enable `use_tiktoken: true` in the SchemaGate config.
- Both within-run (MinHash/Diversity) and cross-run (EmbeddingDeduplicator) deduplication
  are available. Cross-run dedup persists an embedding index across runs; use
  `embedding_reset_index: true` to start fresh.

## Reproduction

```bash
pip install curatorkit
curatorkit run pipeline.yaml --output-dir ./out/
```

## Tool Versions

- **curatorkit**: 0.1.27
- **python**: 3.12.14
