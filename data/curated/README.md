---
license: apache-2.0
pretty_name: ZOMBIE-WEIGHTS Bench
tags:
  - ai-safety
  - unlearning
  - wmdp
  - multilingual
  - canary
---

# ZOMBIE-WEIGHTS Bench

Zero-overlap unlearning benchmark built for the Lexsi AI Safety hackathon.

## Contents

| Group | File | Records |
|---|---|---|
| Arm A forget-train prose | `arm_a/train.jsonl` | 1246 |
| Arm A held-out MCQ | `arm_a/eval_mcq.jsonl` | 3600 |
| Arm B canary records | `arm_b/train.jsonl` | 200 |
| Arm B held-out prompts | `arm_b/eval_prompts.jsonl` | 1000 |
| Global-MMLU controls | `controls/global_mmlu.jsonl` | 900 |
| OR-Bench controls | `controls/or_bench.jsonl` | 300 |
| Benign multilingual | `controls/benign_multilingual.jsonl` | 100 |

Arm A eval: n=300 per task per language (en, hi, hinglish, ar, fr, es), answer index A/B/C/D preserved.
Arm A bio split: strict topic-disjoint, 40.0% train / rest held-out,
Jaccard threshold 0.35.

**Never uploaded:** `cais/wmdp-bio-forget-corpus` raw text. The bio train prose is derived
from the public MIT-licensed `cais/wmdp` MCQ split, not from the gated forget corpus.

Translations: `hf-llm` with per-item validation; translations never alter the
answer index. Hinglish is deterministic Devanagari-to-Roman transliteration of the Hindi twin.
Per-method counts are recorded in `manifest.json` (`counts.arm_a.translation_methods`);
`identity_code` / `identity_refused` entries are explicit, counted fallbacks for code or
safety-filtered segments — never synthetic placeholders.

See `manifest.json` and `lexsi_provenance.json` for hashes, counts, and CuratorKIT run provenance.
