# Dataset Card: arm-a-heldout-eval

## Summary

- **Run timestamp**: 2026-11-13T01:16:26.073262Z
- **Wall clock time**: 0.4s
- **Pipeline config hash**: `9366c9c1a1d10a72`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 3600 | 3600 | — | 0 | 3600 |
| SchemaGate | 3600 | 3600 | 0 | 0 | 3600 |
| TextCleaner | 3600 | 3600 | — | — | 3600 |
| CorpusExporter | — | 3600 | — | — | 3600 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 3600/0 passed (N/A), 0 rejected
- **SchemaGate**: 3600/3600 passed (100.0%), 0 rejected


**Rejection reasons:**
  - None

## Deduplication

  - No deduplication step recorded

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
