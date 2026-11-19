"""Shared schema, normalization, MCQ, and canary helpers for ZOMBIE-WEIGHTS.

All functions are pure and deterministic so the dataset builder and the
pytest/Hypothesis suite exercise exactly the same logic.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any

LANGS: tuple[str, ...] = ("en", "hi", "hinglish", "ar", "fr", "es")
EVAL_LANGS: tuple[str, ...] = ("en", "hi", "hinglish", "ar")
FALLBACK_LANGS: tuple[str, ...] = ("fr", "es")
LABELS: tuple[str, ...] = ("A", "B", "C", "D")

CANARY_RE = re.compile(r"^CANARY-ZW-[A-Z0-9]{4}$")
CANARY_SEARCH_RE = re.compile(r"CANARY-ZW-[A-Z0-9]{4}")
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
CODE_PLACEHOLDER_RE = re.compile(r"__ZWCODE(\d+)__")
FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")

# A conservative stopword list; technical terms are deliberately kept.
_STOPWORDS = frozenset(
    """
    the a an of and or to in for with on by from is are was were be been being
    that which what who whom whose how why when where does do did can could
    should would may might will shall this these those it its as at into over
    under about after before during while than then there their they them he
    she his her not no yes also more most less least such other others using
    used use following answer correct question true false all any each both
    few many some own same so only very just because if but until against
    between through above below up down out off again further once here
    """.split()
)

_SUFFIXES: tuple[str, ...] = (
    "ization", "ations", "ation", "ingly", "edly", "ing", "ers", "er",
    "ies", "ied", "es", "ed", "s",
)


def normalize_text(text: str) -> str:
    """Unicode-normalize and collapse whitespace (idempotent)."""
    if not isinstance(text, str):
        raise TypeError(f"normalize_text expects str, got {type(text)!r}")
    out = unicodedata.normalize("NFKC", text).lower()
    out = re.sub(r"\s+", " ", out)
    return out.strip()


def normalize_for_overlap(text: str) -> str:
    """Aggressive normalization used for zero-leakage checks.

    Strips punctuation (keeping Devanagari/Arabic codepoints and word chars)
    so that trivial formatting differences cannot hide a leaked question.
    """
    out = normalize_text(text)
    out = re.sub(r"[^\w\s\u0900-\u097F\u0600-\u06FF]", " ", out)
    return re.sub(r"\s+", " ", out).strip()


def stem_token(token: str) -> str:
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def topic_signature(text: str, max_tokens: int | None = None) -> frozenset[str]:
    """Deterministic topic signature for the WMDP-bio topic-disjoint split."""
    tokens = re.findall(r"[a-z0-9]+", normalize_text(text))
    sig = {stem_token(t) for t in tokens if len(t) >= 4 and t not in _STOPWORDS}
    if max_tokens is not None:
        return frozenset(sorted(sig)[:max_tokens])
    return frozenset(sig)


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_mcq(
    *,
    question_id: str,
    dataset: str,
    lang: str,
    question: str,
    options: list[str],
    answer_index: int,
    source: str,
    translation: dict[str, Any] | None = None,
    subject: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and construct one unified MCQ record.

    Raises ValueError on any structural violation so that no broken
    A/B/C/D option can ever reach the published dataset.
    """
    if lang not in LANGS:
        raise ValueError(f"unknown lang {lang!r}")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{question_id}/{lang}: empty question")
    if len(options) != 4:
        raise ValueError(f"{question_id}/{lang}: expected 4 options, got {len(options)}")
    if not all(isinstance(o, str) and o.strip() for o in options):
        raise ValueError(f"{question_id}/{lang}: empty option")
    if len({normalize_text(o) for o in options}) != 4:
        raise ValueError(f"{question_id}/{lang}: duplicate options")
    if not 0 <= answer_index <= 3:
        raise ValueError(f"{question_id}/{lang}: answer_index {answer_index} out of range")

    rec: dict[str, Any] = {
        "id": f"{question_id}/{lang}",
        "question_id": question_id,
        "dataset": dataset,
        "lang": lang,
        "question": question,
        "options": list(options),
        "answer_index": int(answer_index),
        "answer_label": LABELS[answer_index],
        "source": source,
        "translation": translation or {"engine": "identity", "src": "en", "method": "identity", "applied": False},
    }
    if subject is not None:
        rec["subject"] = subject
    if extra:
        rec.update(extra)
    return rec


def render_mcq_prompt(rec: dict[str, Any], preamble: str = "") -> str:
    opts = "\n".join(f"{LABELS[i]}. {opt}" for i, opt in enumerate(rec["options"]))
    body = f"Question: {rec['question']}\n{opts}\nAnswer:"
    return f"{preamble}\n\n{body}" if preamble else body


def is_code_like(text: str) -> bool:
    """Heuristic: strings that should not be machine-translated (code / hex).

    WMDP-cyber embeds raw C snippets without markdown fences, so marker-based
    detection is required in addition to the symbol-density check.
    """
    stripped = text.strip()
    if not stripped:
        return False
    if "test(" in stripped or stripped.startswith("0x"):
        return True
    markers = (
        "#include", "#define", "#pragma", "int main", "printf(", "return ",
        "unsigned ", "signed ", "char ", "long long", "void ", "static ",
        "const ", "struct ", "uint", "int8_t", "int16_t", "int32_t", "int64_t",
        "//", "/*", "*/", "sizeof(", "malloc(", "->",
        # x86 / assembly (WMDP-cyber embeds raw asm blocks)
        "x86", "assembly", "intel syntax", "mov ", "xor ", "jmp ", "lea ",
        "push ", "pop ", "rax", "rbx", "rcx", "rdx", "eax", "rsp", "rbp",
        "call ", "cmp ", "add ", "sub ",
        # shell / offensive-tool commands
        "msfpayload", "msfconsole", "meterpreter", "lhost=", "lport=",
        "reverse_tcp", "powershell", "cmd.exe", "bash ", "curl ", "wget ",
        "nmap ", "netcat", "nc -", "python -c", "chmod ", "sudo ",
        "base64 ", "eval(", "exiftool", "reg add", "schtasks", "wmic ",
    )
    marker_hits = sum(1 for marker in markers if marker in stripped)
    if marker_hits >= 2:
        return True
    letters = sum(ch.isalpha() for ch in stripped)
    if letters < 3:
        return True
    symbols = sum(ch in "{}()[];=<>/*&|!0123456789'\"\\" for ch in stripped)
    return symbols / max(len(stripped), 1) > 0.28


def protect_code_spans(text: str) -> tuple[str, list[str]]:
    """Replace fenced/inline code with stable placeholders before translation."""
    spans: list[str] = []

    def _sub(match: re.Match[str]) -> str:
        spans.append(match.group(0))
        return f"__ZWCODE{len(spans) - 1}__"

    out = FENCED_CODE_RE.sub(_sub, text)
    out = INLINE_CODE_RE.sub(_sub, out)
    return out, spans


def restore_code_spans(text: str, spans: list[str]) -> str:
    def _sub(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        return spans[idx] if idx < len(spans) else match.group(0)

    return CODE_PLACEHOLDER_RE.sub(_sub, text)


def make_canary(rng) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "CANARY-ZW-" + "".join(rng.choice(alphabet) for _ in range(4))


def validate_canary(value: str) -> bool:
    return bool(CANARY_RE.match(value))


def find_canaries(text: str) -> list[str]:
    return CANARY_SEARCH_RE.findall(text)
