# Dataset Card: arm-b-canary-train

## Summary

- **Run timestamp**: 2026-11-13T01:16:26.149324Z
- **Wall clock time**: 0.0s
- **Pipeline config hash**: `acf770d5a0f1ecdb`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 200 | 200 | — | 0 | 200 |
| SchemaGate | 200 | 200 | 0 | 0 | 200 |
| ExactDeduplicator | 200 | 200 | — | — | 200 |
| TextCleaner | 200 | 200 | — | — | 200 |
| CorpusExporter | — | 200 | — | — | 200 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 200/0 passed (N/A), 0 rejected
- **SchemaGate**: 200/200 passed (100.0%), 0 rejected


**Rejection reasons:**
  - None

## Deduplication

  - exact_duplicates_removed: 0

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
