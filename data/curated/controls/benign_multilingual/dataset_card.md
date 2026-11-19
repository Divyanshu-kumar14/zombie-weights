# Dataset Card: controls-benign-multilingual

## Summary

- **Run timestamp**: 2026-11-13T01:16:58.670487Z
- **Wall clock time**: 0.0s
- **Pipeline config hash**: `5698c69bf0ed4ac5`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 100 | 100 | — | 0 | 100 |
| SchemaGate | 100 | 100 | 0 | 0 | 100 |
| TextCleaner | 100 | 100 | — | — | 100 |
| CorpusExporter | — | 100 | — | — | 100 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 100/0 passed (N/A), 0 rejected
- **SchemaGate**: 100/100 passed (100.0%), 0 rejected


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
