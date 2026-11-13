"""Verification suite for the ZOMBIE-WEIGHTS Phase 0/1 dataset build.

Run after ``python -m src.data.build_datasets``:

    .venv/bin/pytest tests/ -v

The suite enforces the hackathon's hard invariants:
  * zero train/eval prompt leakage (Arm A + Arm B)
  * answer-index invariance across every language
  * strict bio topic-disjointness
  * valid CANARY-ZW-XXXX formatting and no canary inside held-out prompts
  * CuratorKIT manifest + provenance presence and checksum integrity
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from src.data.build_datasets import (
    BIO_TRAIN_FRACTION,
    LANGS,
    N_CANARY_RECORDS,
    TOPIC_JACCARD_THRESHOLD,
    containment_report,
    load_wmdp,
    topic_disjoint_split,
    validate_outputs,
)
from src.data.schema import (
    CANARY_RE,
    LABELS,
    build_mcq,
    jaccard,
    normalize_for_overlap,
    normalize_text,
    render_mcq_prompt,
    sha256_file,
    topic_signature,
    validate_canary,
)
from src.data.translate import TranslationError, transliterate_hinglish, validate_translation

DATA = Path(__file__).resolve().parent.parent / "data" / "curated"

pytestmark = pytest.mark.skipif(not DATA.exists(), reason="run src.data.build_datasets first")


def _jsonl(name: str) -> list[dict]:
    path = DATA / name
    assert path.exists(), f"missing artifact {path}"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Counts / structure
# ---------------------------------------------------------------------------


def test_arm_a_counts():
    eval_a = _jsonl("arm_a/eval_mcq.jsonl")
    assert len(eval_a) == 2 * 300 * len(LANGS)
    for lang in LANGS:
        for task in ("wmdp-cyber", "wmdp-bio"):
            n = sum(1 for r in eval_a if r["lang"] == lang and r["dataset"] == task)
            assert n == 300, f"{task}/{lang} has {n} records"


def test_arm_b_counts():
    train_b = _jsonl("arm_b/train.jsonl")
    eval_b = _jsonl("arm_b/eval_prompts.jsonl")
    assert len(train_b) == N_CANARY_RECORDS
    assert len(eval_b) == N_CANARY_RECORDS * 5
    assert {p["variant"] for p in eval_b} == {"direct", "json", "hinglish", "hindi", "arabic"}


def test_controls_counts():
    mmlu = _jsonl("controls/global_mmlu.jsonl")
    or_bench = _jsonl("controls/or_bench.jsonl")
    benign = _jsonl("controls/benign_multilingual.jsonl")
    assert len(mmlu) == 900
    assert {lang: sum(1 for r in mmlu if r["lang"] == lang) for lang in ("en", "hi", "ar")} == {"en": 300, "hi": 300, "ar": 300}
    assert len(or_bench) == 300
    assert len(benign) == 100
    assert {r["lang"] for r in benign} == {"hinglish", "ar"}


def test_mcq_structure_and_unique_ids():
    records = _jsonl("arm_a/eval_mcq.jsonl") + _jsonl("controls/global_mmlu.jsonl")
    ids = set()
    for rec in records:
        assert rec["id"] not in ids, f"duplicate id {rec['id']}"
        ids.add(rec["id"])
        assert len(rec["options"]) == 4
        assert len({normalize_text(o) for o in rec["options"]}) == 4
        assert rec["answer_index"] in (0, 1, 2, 3)
        assert rec["answer_label"] == LABELS[rec["answer_index"]]
        assert rec["question"].strip()
        assert all(o.strip() for o in rec["options"])
        prompt = render_mcq_prompt(rec)
        for label in LABELS:
            assert f"{label}." in prompt


def test_answer_index_invariance_across_languages():
    records = _jsonl("arm_a/eval_mcq.jsonl")
    by_question: dict[str, list[dict]] = {}
    for rec in records:
        by_question.setdefault(rec["question_id"], []).append(rec)
    assert len(by_question) == 600
    for qid, group in by_question.items():
        langs = {r["lang"] for r in group}
        assert langs == set(LANGS), f"{qid} missing languages: {set(LANGS) - langs}"
        answers = {r["answer_index"] for r in group}
        assert len(answers) == 1, f"{qid} answer index changed across languages: {answers}"
        assert all(len(r["options"]) == 4 for r in group)


def test_translation_scripts_and_no_empties():
    records = _jsonl("arm_a/eval_mcq.jsonl")
    for rec in records:
        lang = rec["lang"]
        method = rec["translation"].get("method")
        if method in ("identity_code", "identity_fallback", "identity_refused"):
            continue
        if lang == "hi":
            assert re.search(r"[\u0900-\u097F]", rec["question"]), rec["id"]
        if lang == "ar":
            assert re.search(r"[\u0600-\u06FF]", rec["question"]), rec["id"]
        if lang == "hinglish":
            assert not re.search(r"[\u0900-\u097F]", rec["question"]), rec["id"]
        for opt in rec["options"]:
            assert opt.strip(), rec["id"]


# ---------------------------------------------------------------------------
# Zero leakage
# ---------------------------------------------------------------------------


def test_zero_leakage_arm_a():
    train_a = [r["text"] for r in _jsonl("arm_a/train.jsonl")]
    eval_a = _jsonl("arm_a/eval_mcq.jsonl")
    hits = containment_report(train_a, [(r["id"], r["question"]) for r in eval_a])
    assert hits == [], f"Arm A eval questions found in train prose: {hits[:5]}"


def test_zero_leakage_arm_b():
    train_b = [r["text"] for r in _jsonl("arm_b/train.jsonl")]
    eval_b = _jsonl("arm_b/eval_prompts.jsonl")
    hits = containment_report(train_b, [(r["id"], r["prompt"]) for r in eval_b])
    assert hits == [], f"Arm B held-out prompts found in canary train prose: {hits[:5]}"


def test_canary_prompts_never_contain_a_canary():
    eval_b = _jsonl("arm_b/eval_prompts.jsonl")
    train_canaries = {r["canary"] for r in _jsonl("arm_b/train.jsonl")}
    for rec in eval_b:
        assert not re.search(r"CANARY-ZW-[A-Z0-9]{4}", rec["prompt"]), rec["id"]
        assert rec["target_canary"] not in rec["prompt"], rec["id"]
        assert rec["target_canary"] in train_canaries


def test_controls_do_not_overlap_arm_a():
    train_a = {normalize_for_overlap(r["text"]) for r in _jsonl("arm_a/train.jsonl")}
    eval_a = {normalize_for_overlap(r["question"]) for r in _jsonl("arm_a/eval_mcq.jsonl")}
    known = train_a | eval_a
    for name, key in (("controls/global_mmlu.jsonl", "question"), ("controls/or_bench.jsonl", "prompt"), ("controls/benign_multilingual.jsonl", "prompt")):
        for rec in _jsonl(name):
            assert normalize_for_overlap(rec[key]) not in known, f"{name}: {rec['id']} duplicates Arm A"


def test_topic_disjointness_is_reproducible_and_strict():
    bio_rows, _ = load_wmdp()
    train_idx, eval_idx, stats = topic_disjoint_split(bio_rows)
    assert stats["threshold"] == TOPIC_JACCARD_THRESHOLD
    assert abs(stats["actual_train_fraction"] - BIO_TRAIN_FRACTION) < 0.02
    assert set(train_idx).isdisjoint(eval_idx)
    assert len(train_idx) + len(eval_idx) == len(bio_rows)

    train_sigs = [topic_signature(bio_rows[i]["question"]) for i in train_idx]
    eval_sigs = [topic_signature(bio_rows[i]["question"]) for i in eval_idx]
    worst = 0.0
    for esig in eval_sigs:
        for tsig in train_sigs:
            worst = max(worst, jaccard(esig, tsig))
    assert worst < TOPIC_JACCARD_THRESHOLD, f"topic-disjointness violated: max Jaccard {worst}"
    assert stats["n_eval_pool"] >= 300


def test_validate_outputs_passes():
    validate_outputs(DATA, {})


# ---------------------------------------------------------------------------
# Canary format
# ---------------------------------------------------------------------------


def test_canary_regex_and_uniqueness():
    records = _jsonl("arm_b/train.jsonl")
    canaries = [r["canary"] for r in records]
    assert len(canaries) == len(set(canaries)) == N_CANARY_RECORDS
    for canary in canaries:
        assert validate_canary(canary), canary
        assert CANARY_RE.match(canary)


# ---------------------------------------------------------------------------
# Manifests / provenance
# ---------------------------------------------------------------------------


def test_curatorkit_manifests_and_provenance_exist():
    for rel in (
        "arm_a/train",
        "arm_a/eval",
        "arm_b/train",
        "arm_b/eval",
        "controls/global_mmlu",
        "controls/or_bench",
        "controls/benign_multilingual",
    ):
        base = DATA / rel
        assert (base / "manifest.json").exists(), rel
        assert (base / "lexsi_provenance.json").exists(), rel


def test_master_manifest_checksums():
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    provenance = json.loads((DATA / "lexsi_provenance.json").read_text(encoding="utf-8"))
    assert manifest["project"] == "zombie-weights"
    assert provenance["schema"].startswith("lexsi_provenance/")
    assert manifest["files"]
    for item in manifest["files"]:
        path = DATA / item["path"]
        assert path.exists(), item["path"]
        assert sha256_file(path) == item["sha256"], item["path"]


def test_provenance_excludes_gated_bio_forget_corpus():
    provenance = json.loads((DATA / "lexsi_provenance.json").read_text(encoding="utf-8"))
    upload_sources = [s["repo"] for s in provenance["sources"]]
    assert "cais/wmdp-bio-forget-corpus" not in upload_sources
    assert not any("bio-forget" in s for s in upload_sources)


# ---------------------------------------------------------------------------
# Hypothesis properties
# ---------------------------------------------------------------------------


@given(st.text(min_size=0, max_size=300))
@settings(max_examples=100, deadline=None)
def test_normalize_text_idempotent(text: str):
    once = normalize_text(text)
    assert normalize_text(once) == once


@given(st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", min_size=4, max_size=4))
@settings(max_examples=100, deadline=None)
def test_canary_suffix_validity(suffix: str):
    assert CANARY_RE.match(f"CANARY-ZW-{suffix}")


@given(st.text(min_size=5, max_size=40).filter(lambda s: s.strip()))
@settings(max_examples=50, deadline=None)
def test_invalid_canary_suffix_rejected(suffix: str):
    if re.fullmatch(r"[A-Z0-9]{4}", suffix):
        return
    assert not CANARY_RE.match(f"CANARY-ZW-{suffix}")


@given(
    st.lists(st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=4, max_size=12), min_size=4, max_size=4, unique=True),
    st.integers(min_value=0, max_value=3),
)
@settings(max_examples=100, deadline=None)
def test_build_mcq_preserves_answer_index(options: list[str], answer_index: int):
    rec = build_mcq(
        question_id="prop/test",
        dataset="synthetic",
        lang="en",
        question="Which option is correct?",
        options=options,
        answer_index=answer_index,
        source="hypothesis",
    )
    assert rec["answer_index"] == answer_index
    assert rec["answer_label"] == LABELS[answer_index]
    assert rec["options"][answer_index] == options[answer_index]


@given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz ", min_size=40, max_size=80))
@settings(max_examples=50, deadline=None)
def test_containment_detects_injected_question(question: str):
    norm = normalize_for_overlap(question)
    if len(norm) < 40:
        return
    train = ["Some unrelated curriculum prose about enterprise security and directory services." * 3]
    assert containment_report(train, [("probe", question)]) == []
    train_with_leak = train + [f"leaked: {question} appended here"]
    assert containment_report(train_with_leak, [("probe", question)])


@given(st.sampled_from(["hello world", "नमस्ते दुनिया", "مرحبا بالعالم", "shared passwords"]))
@settings(max_examples=20, deadline=None)
def test_translation_validation_contract(text: str):
    with pytest.raises(TranslationError):
        validate_translation(text, "", "hi")
    with pytest.raises(TranslationError):
        validate_translation(
            "hello world this is an untranslated sentence",
            "hello world this is an untranslated sentence",
            "fr",
        )
    if text.strip():
        validate_translation(text, text, "ar" if re.search(r"[\u0600-\u06FF]", text) else "en")


def test_hinglish_transliteration_is_roman_and_deterministic():
    hi = "यह वायरस श्वास के ज़रिए फैलता है।"
    once = transliterate_hinglish(hi)
    twice = transliterate_hinglish(hi)
    assert once == twice
    assert once.strip()
    assert not re.search(r"[\u0900-\u097F]", once)
