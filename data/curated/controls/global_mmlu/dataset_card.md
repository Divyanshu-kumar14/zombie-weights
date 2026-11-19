# Dataset Card: controls-global-mmlu

## Summary

- **Run timestamp**: 2026-11-13T01:16:58.651416Z
- **Wall clock time**: 0.2s
- **Pipeline config hash**: `8943769e3491173f`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 900 | 900 | — | 0 | 900 |
| SchemaGate | 900 | 900 | 0 | 0 | 900 |
| TextCleaner | 900 | 900 | — | — | 900 |
| CorpusExporter | — | 900 | — | — | 900 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 900/0 passed (N/A), 0 rejected
- **SchemaGate**: 900/900 passed (100.0%), 0 rejected


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
