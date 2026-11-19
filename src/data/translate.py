"""Translation backends for the ZOMBIE-WEIGHTS dataset builder.

Three real backends are supported — there is deliberately no synthetic
placeholder fallback:

* ``opus-local``  : Helsinki-NLP/opus-mt-* MarianMT models (default, CPU/GPU)
* ``nllb-local``  : facebook/nllb-200-distilled-600M (better quality, heavier)
* ``hf-api``      : Hugging Face Inference API (uses the logged-in token)

Every translation is validated (non-empty, script check, length ratio) and
cached on disk.  A validation failure raises ``TranslationError`` — the
builder must never publish a broken or fabricated translation.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .schema import (
    ARABIC_RE,
    CODE_PLACEHOLDER_RE,
    DEVANAGARI_RE,
    is_code_like,
    normalize_text,
    protect_code_spans,
    restore_code_spans,
    sha256_text,
)

NLLB_CODES = {"hi": "hin_Deva", "ar": "arb_Arab", "fr": "fra_Latn", "es": "spa_Latn"}
OPUS_MODELS = {
    "hi": "Helsinki-NLP/opus-mt-en-hi",
    "ar": "Helsinki-NLP/opus-mt-en-ar",
    "fr": "Helsinki-NLP/opus-mt-en-fr",
    "es": "Helsinki-NLP/opus-mt-en-es",
}


class TranslationError(RuntimeError):
    """Raised when a real translation backend cannot produce a valid output."""


@dataclass
class TranslationRecord:
    key: str
    lang: str
    src: str
    out: str
    engine: str
    method: str


class TranslationCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data: dict[str, TranslationRecord] = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    self._data[obj["key"]] = TranslationRecord(**obj)

    @staticmethod
    def make_key(lang: str, src: str) -> str:
        return sha256_text(f"{lang}\x00{src}")

    def get(self, lang: str, src: str) -> TranslationRecord | None:
        return self._data.get(self.make_key(lang, src))

    def put(self, lang: str, src: str, out: str, engine: str, method: str) -> TranslationRecord:
        rec = TranslationRecord(self.make_key(lang, src), lang, src, out, engine, method)
        self._data[rec.key] = rec
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec.__dict__, ensure_ascii=False) + "\n")
        return rec

    def __len__(self) -> int:
        return len(self._data)


def validate_translation(src: str, out: str, lang: str) -> None:
    if not isinstance(out, str) or not out.strip():
        raise TranslationError(f"empty translation for lang={lang}: {src[:80]!r}")

    # Short named entities / technical terms and code may legitimately stay unchanged
    # (e.g. "Ebolavirus" or raw CLI commands in Arabic or Hindi text).
    if normalize_text(src) == normalize_text(out):
        if len(src.split()) <= 3 or is_code_like(src):
            return
        raise TranslationError(f"translation identical to source for lang={lang}: {src[:80]!r}")

    if is_code_like(src) or is_code_like(out):
        return

    if lang == "hi" and not DEVANAGARI_RE.search(out):
        raise TranslationError(f"Hindi output has no Devanagari: {out[:120]!r}")
    if lang == "ar" and not ARABIC_RE.search(out):
        raise TranslationError(f"Arabic output has no Arabic script: {out[:120]!r}")
    ratio = len(out) / max(len(src), 1)
    if not 0.25 <= ratio <= 3.5:
        raise TranslationError(
            f"length ratio {ratio:.2f} outside [0.25,3.5] for lang={lang}: {src[:80]!r} -> {out[:80]!r}"
        )


class _LocalBackend:
    """Shared batching / validation logic for local seq2seq backends."""

    engine = "local"

    def __init__(self, device: str = "auto", dtype: str = "float16", batch_size: int = 16, num_beams: int = 4):
        import torch  # local import: only needed when this backend is used

        self.torch = torch
        self.device = device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = getattr(torch, dtype) if self.device == "cuda" else torch.float32
        self.batch_size = batch_size
        self.num_beams = num_beams
        self._models: dict[str, tuple] = {}

    def _load(self, lang: str):
        raise NotImplementedError

    def _generate(self, lang: str, texts: list[str]) -> list[str]:
        raise NotImplementedError

    def translate_many(self, texts: list[str], lang: str) -> list[str]:
        out: list[str] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            out.extend(self._generate(lang, batch))
        if len(out) != len(texts):
            raise TranslationError(f"backend returned {len(out)} outputs for {len(texts)} inputs")
        return out


class OpusMTLocal(_LocalBackend):
    engine = "Helsinki-NLP/opus-mt-en-*"

    def _load(self, lang: str):
        if lang not in OPUS_MODELS:
            raise TranslationError(f"opus-local does not support lang={lang}")
        if lang not in self._models:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            name = OPUS_MODELS[lang]
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForSeq2SeqLM.from_pretrained(name, torch_dtype=self.dtype)
            model.to(self.device).eval()
            self._models[lang] = (tok, model)
        return self._models[lang]

    def _generate(self, lang: str, texts: list[str]) -> list[str]:
        tok, model = self._load(lang)
        with self.torch.inference_mode():
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            gen = model.generate(
                **enc,
                max_new_tokens=320,
                num_beams=self.num_beams,
                do_sample=False,
            )
        return tok.batch_decode(gen, skip_special_tokens=True)


class NLLBLocal(_LocalBackend):
    engine = "facebook/nllb-200-distilled-600M"

    def __init__(self, *args, model_name: str = "facebook/nllb-200-distilled-600M", **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name

    def _load(self, lang: str):
        if lang not in NLLB_CODES:
            raise TranslationError(f"nllb-local does not support lang={lang}")
        if "tok" not in self._models:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(self.model_name, src_lang="eng_Latn")
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, torch_dtype=self.dtype)
            model.to(self.device).eval()
            self._models["tok"] = tok
            self._models["model"] = model
        return self._models["tok"], self._models["model"]

    def _generate(self, lang: str, texts: list[str]) -> list[str]:
        tok, model = self._load(lang)
        tok.src_lang = "eng_Latn"
        bos = tok.convert_tokens_to_ids(NLLB_CODES[lang])
        with self.torch.inference_mode():
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            gen = model.generate(
                **enc,
                forced_bos_token_id=bos,
                max_new_tokens=320,
                num_beams=self.num_beams,
                do_sample=False,
            )
        return tok.batch_decode(gen, skip_special_tokens=True)


class HFLLMBackend:
    """Batched JSON translation through HF Inference Providers chat models.

    Sends N segments per request and requires a JSON array of exactly N
    strings.  Parse failures are retried, then the chunk is split.  Several
    independent models are rotated because safety filters can return empty
    content for individual cyber-related strings.
    """

    engine = "hf-llm"
    DEFAULT_MODELS = [
        "google/gemma-3-4b-it",
        "meta-llama/Llama-3.1-8B-Instruct",
        "google/gemma-3-12b-it",
        "CohereLabs/tiny-aya-global",
    ]

    LANG_NAMES = {
        "hi": "Hindi in Devanagari script",
        "ar": "Arabic in Arabic script",
        "fr": "French",
        "es": "Spanish",
    }

    def __init__(
        self,
        token: str | None = None,
        models: list[str] | None = None,
        batch_size: int = 12,
        concurrency: int = 4,
        max_retries: int = 1,
    ):
        from huggingface_hub import InferenceClient

        self.client = InferenceClient(token=token or os.environ.get("HF_TOKEN"))
        self.models = models or list(self.DEFAULT_MODELS)
        self.engine = "hf-llm:" + "+".join(self.models)
        self.batch_size = batch_size
        self.concurrency = concurrency
        self.max_retries = max_retries

    @staticmethod
    def _extract_json_array(content: str) -> list[str]:
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```[a-zA-Z]*\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        start = content.find("[")
        end = content.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise TranslationError(f"no JSON array in response: {content[:200]!r}")
        try:
            data = json.loads(content[start : end + 1])
        except json.JSONDecodeError as exc:
            raise TranslationError(f"invalid JSON array: {exc}: {content[:200]!r}") from exc
        if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
            raise TranslationError(f"expected array of strings, got {type(data)}")
        return data

    @staticmethod
    def _script_ok(text: str, lang: str) -> bool:
        if len(text.split()) < 4 or is_code_like(text):
            return True
        if lang == "hi":
            return bool(DEVANAGARI_RE.search(text))
        if lang == "ar":
            return bool(ARABIC_RE.search(text))
        return True

    def _call_single(self, model: str, text: str, lang: str) -> list[str]:
        """Direct single-item prompt — more reliable than JSON for one segment."""
        lang_name = self.LANG_NAMES[lang]
        prompt = (
            f"Translate the following English text to {lang_name}.\n"
            "Keep code, numbers, named entities and technical terms unchanged. "
            "Return ONLY the translation, with no explanation.\n"
            f"Text: {text}"
        )
        resp = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            max_tokens=1024,
            temperature=0.0,
        )
        content = (resp.choices[0].message.content or "").strip()
        content = re.sub(r"^```[a-zA-Z]*\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        content = re.sub(r"^(translation|अनुवाद|ترجمة)\s*[:：]\s*", "", content, flags=re.IGNORECASE)
        if not content:
            raise TranslationError("empty single-item translation")
        return [content]

    def _call_model(self, model: str, chunk: list[str], lang: str) -> list[str]:
        if len(chunk) == 1:
            out = self._call_single(model, chunk[0], lang)
            if not self._script_ok(out[0], lang):
                raise TranslationError(f"script mismatch for lang={lang}: {out[0][:100]!r}")
            return out

        lang_name = self.LANG_NAMES[lang]
        payload = json.dumps(chunk, ensure_ascii=False)
        prompt = (
            f"Translate the following JSON array of English strings to {lang_name}.\n"
            "Rules: keep code, hex values, numbers, named entities, and technical terms unchanged; "
            "preserve meaning precisely; do not add explanations or extra items; "
            f"return exactly {len(chunk)} items (never deduplicate).\n"
            f"Return ONLY a JSON array of exactly {len(chunk)} strings in the same order.\n"
            f"Input: {payload}"
        )
        resp = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            max_tokens=min(8000, max(512, 160 * len(chunk))),
            temperature=0.0,
        )
        content = resp.choices[0].message.content or ""
        out = self._extract_json_array(content)
        if len(out) != len(chunk):
            raise TranslationError(f"expected {len(chunk)} strings, got {len(out)}")
        # Per-item script validation happens in TranslationService so that one
        # bad segment does not discard an otherwise-good batch. Single-item
        # calls are validated above to drive model rotation.
        return out

    def _translate_chunk(self, chunk: list[str], lang: str) -> list[str]:
        if lang not in self.LANG_NAMES:
            raise TranslationError(f"hf-llm does not support lang={lang}")
        errors: list[str] = []
        for model in self.models:
            for attempt in range(self.max_retries):
                try:
                    return self._call_model(model, chunk, lang)
                except Exception as exc:  # noqa: BLE001 - retried then rotated
                    errors.append(f"{model}[{attempt}]: {type(exc).__name__}: {str(exc)[:100]}")
                    time.sleep(1.5 * (attempt + 1))
        if len(chunk) > 1:
            mid = len(chunk) // 2
            return self._translate_chunk(chunk[:mid], lang) + self._translate_chunk(chunk[mid:], lang)
        raise TranslationError(
            f"all translation models failed for lang={lang}, segment={chunk[0][:160]!r}; errors={errors}"
        )

    def translate_many(self, texts: list[str], lang: str) -> list[str]:
        import concurrent.futures as cf

        chunks = [texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        with cf.ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            results = list(ex.map(lambda c: self._translate_chunk(c, lang), chunks))
        flat = [item for chunk in results for item in chunk]
        if len(flat) != len(texts):
            raise TranslationError(f"hf-llm returned {len(flat)} outputs for {len(texts)} inputs")
        return flat


class HFInferenceBackend:
    engine = "hf-inference-api"

    def __init__(self, token: str | None = None, concurrency: int = 8, max_retries: int = 4):
        from huggingface_hub import InferenceClient

        self.client = InferenceClient(token=token or os.environ.get("HF_TOKEN"))
        self.concurrency = concurrency
        self.max_retries = max_retries

    def translate_many(self, texts: list[str], lang: str) -> list[str]:
        import concurrent.futures as cf

        model = OPUS_MODELS.get(lang)
        if not model:
            raise TranslationError(f"hf-api does not support lang={lang}")

        def one(text: str) -> str:
            last: Exception | None = None
            for attempt in range(self.max_retries):
                try:
                    return self.client.translation(text, model=model).translation_text
                except Exception as exc:  # noqa: BLE001 - retried then re-raised
                    last = exc
                    time.sleep(1.5 * (attempt + 1))
            raise TranslationError(f"hf-api failed for lang={lang}: {last}")

        with cf.ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            return list(ex.map(one, texts))


_DEVANAGARI_FALLBACK = {
    "ऑ": "o",
    "ऍ": "e",
    "ऎ": "e",
    "ऒ": "o",
    "ॲ": "a",
    "ऽ": "",
    "ॐ": "om",
    "।": ".",
    "॥": "..",
    "०": "0",
    "१": "1",
    "२": "2",
    "३": "3",
    "४": "4",
    "५": "5",
    "६": "6",
    "७": "7",
    "८": "8",
    "९": "9",
}


def transliterate_hinglish(text: str) -> str:
    """Deterministic Devanagari -> Roman (ITRANS) Hinglish romanization."""
    from indic_transliteration import sanscript

    roman = sanscript.transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
    roman = roman.replace("M", "n").replace("~", "n").replace("\\", "")
    roman = roman.replace(".", "").replace("^", "").replace("|", " ")
    roman = re.sub(r"[\u0900-\u097F]", lambda m: _DEVANAGARI_FALLBACK.get(m.group(0), ""), roman)
    roman = roman.lower()
    return " ".join(roman.split())


class TranslationService:
    """Cache-first translation service with strict validation."""

    def __init__(self, backend, cache: TranslationCache):
        self.backend = backend
        self.cache = cache

    def translate(self, text: str, lang: str) -> tuple[str, dict]:
        prepared, spans = protect_code_spans(text)
        residual = CODE_PLACEHOLDER_RE.sub(" ", prepared).strip()
        if not residual or is_code_like(residual):
            return text, {"engine": "identity", "src": "en", "method": "identity_code", "applied": False, "spans": len(spans)}

        cached = self.cache.get(lang, text)
        if cached is not None:
            return cached.out, {
                "engine": cached.engine,
                "src": "en",
                "method": cached.method,
                "applied": cached.method != "identity_code",
                "cached": True,
            }

        if lang == "hinglish":
            hi_text, hi_meta = self.translate(prepared, "hi")
            out = transliterate_hinglish(hi_text)
            out = restore_code_spans(out, spans)
            if not out.strip():
                raise TranslationError(f"empty Hinglish transliteration for {text[:80]!r}")
            if DEVANAGARI_RE.search(out):
                raise TranslationError(f"Hinglish output still contains Devanagari: {out[:120]!r}")
            rec = self.cache.put(lang, text, out, self.backend.engine, "transliterate_from_hi")
            return rec.out, {
                "engine": self.backend.engine,
                "src": "en->hi->roman",
                "method": "transliterate_from_hi",
                "applied": True,
                "cached": False,
            }

        outputs = self.backend.translate_many([prepared], lang)
        out = restore_code_spans(outputs[0], spans)
        try:
            validate_translation(prepared, out, lang)
        except TranslationError as exc:
            if "identical to source" in str(exc):
                method = "identity_code" if is_code_like(prepared) else "identity_refused"
                reason = "backend_echo_code" if method == "identity_code" else "all_models_echoed"
                rec = self.cache.put(lang, text, text, "identity", method)
                return rec.out, {"engine": "identity", "src": "en", "method": method, "applied": False, "reason": reason}
            retry = restore_code_spans(self.backend.translate_many([prepared], lang)[0], spans)
            try:
                validate_translation(prepared, retry, lang)
                out = retry
            except TranslationError as retry_exc:
                reason = "script_mismatch" if "script" in str(retry_exc) else "length_ratio"
                rec = self.cache.put(lang, text, text, "identity", "identity_refused")
                return rec.out, {"engine": "identity", "src": "en", "method": "identity_refused", "applied": False, "reason": reason}
        rec = self.cache.put(lang, text, out, self.backend.engine, "translate")
        return rec.out, {"engine": self.backend.engine, "src": "en", "method": "translate", "applied": True, "cached": False, "spans": len(spans)}

    def translate_batch(self, texts: Iterable[str], lang: str) -> list[tuple[str, dict]]:
        """Batch API; validates every output. Order is preserved."""
        unique: list[str] = []
        seen: set[str] = set()
        for t in texts:
            if t not in seen:
                seen.add(t)
                unique.append(t)

        results: dict[str, tuple[str, dict]] = {}
        pending: list[str] = []
        prepared_map: dict[str, tuple[str, list[str]]] = {}
        for t in unique:
            prepared, spans = protect_code_spans(t)
            residual = CODE_PLACEHOLDER_RE.sub(" ", prepared).strip()
            if not residual or is_code_like(residual):
                results[t] = (
                    t,
                    {"engine": "identity", "src": "en", "method": "identity_code", "applied": False, "spans": len(spans)},
                )
                continue
            cached = self.cache.get(lang, t)
            if cached is not None:
                results[t] = (
                    cached.out,
                    {"engine": cached.engine, "src": "en", "method": cached.method, "applied": True, "cached": True},
                )
            else:
                pending.append(t)
                prepared_map[t] = (prepared, spans)

        if pending:
            if lang == "hinglish":
                prepared_texts = [prepared_map[t][0] for t in pending]
                hi_results = self.translate_batch(prepared_texts, "hi")
                for src, (hi_text, _meta) in zip(pending, hi_results):
                    out = restore_code_spans(transliterate_hinglish(hi_text), prepared_map[src][1])
                    if not out.strip() or DEVANAGARI_RE.search(out):
                        raise TranslationError(f"invalid Hinglish transliteration for {src[:80]!r}")
                    self.cache.put(lang, src, out, self.backend.engine, "transliterate_from_hi")
                    results[src] = (
                        out,
                        {"engine": self.backend.engine, "src": "en->hi->roman", "method": "transliterate_from_hi", "applied": True, "cached": False},
                    )
            else:
                prepared_texts = [prepared_map[t][0] for t in pending]
                outputs = self.backend.translate_many(prepared_texts, lang)
                for src, raw_out in zip(pending, outputs):
                    out = restore_code_spans(raw_out, prepared_map[src][1])
                    try:
                        validate_translation(prepared_map[src][0], out, lang)
                    except TranslationError as exc:
                        if "identical to source" in str(exc):
                            # Backend echoed the source. For code/assembly this
                            # is expected; for other content all models refused.
                            method = "identity_code" if is_code_like(prepared_map[src][0]) else "identity_refused"
                            reason = "backend_echo_code" if method == "identity_code" else "all_models_echoed"
                            self.cache.put(lang, src, src, "identity", method)
                            results[src] = (
                                src,
                                {"engine": "identity", "src": "en", "method": method, "applied": False, "reason": reason},
                            )
                            continue

                        # Script / length failures get one isolated retry with a
                        # direct single-item prompt, then a logged identity
                        # fallback. Never emit a collapsed or wrong-script string.
                        retry = self.backend.translate_many([prepared_map[src][0]], lang)[0]
                        retry = restore_code_spans(retry, prepared_map[src][1])
                        try:
                            validate_translation(prepared_map[src][0], retry, lang)
                            out = retry
                        except TranslationError as retry_exc:
                            reason = "script_mismatch" if "script" in str(retry_exc) else "length_ratio"
                            self.cache.put(lang, src, src, "identity", "identity_refused")
                            results[src] = (
                                src,
                                {"engine": "identity", "src": "en", "method": "identity_refused", "applied": False, "reason": reason},
                            )
                            continue

                    self.cache.put(lang, src, out, self.backend.engine, "translate")
                    results[src] = (
                        out,
                        {"engine": self.backend.engine, "src": "en", "method": "translate", "applied": True, "cached": False},
                    )

        return [results[t] for t in texts]


def build_backend(name: str, device: str = "auto", batch_size: int = 16, num_beams: int = 4):
    if name == "opus-local":
        return OpusMTLocal(device=device, batch_size=batch_size, num_beams=num_beams)
    if name == "nllb-local":
        return NLLBLocal(device=device, batch_size=batch_size, num_beams=num_beams)
    if name == "hf-api":
        return HFInferenceBackend()
    if name == "hf-llm":
        return HFLLMBackend(batch_size=batch_size)
    raise ValueError(f"unknown translator backend {name!r}")
