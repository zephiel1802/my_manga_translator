"""Small translation-stage models and config helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


STYLE_PROMPTS: dict[str, str] = {
    "Default": "",
    "Casual": "Use casual, natural everyday language that sounds like spoken dialogue.",
    "Formal": "Use formal, polite language while keeping the dialogue natural.",
    "Keep Honorifics": "Keep Japanese honorifics like -san, -kun, -chan, -sama, senpai, and sensei.",
    "Web Novel Style": "Use dramatic web novel style with emotional weight and vivid phrasing.",
    "Action": "Use short, punchy lines with fast pacing and strong impact.",
    "Literal": "Stay close to the original meaning while remaining readable and natural.",
    "Custom": "",
}

TRANSLATOR_CHOICES = (
    "Gemini",
    "Local LLM",
    "DeepSeek",
    "Google",
    "NLLB",
    "Baidu",
    "Bing",
)

LANGUAGE_CHOICES = (
    "ja",
    "en",
    "vi",
    "zh",
    "ko",
    "th",
    "id",
    "fr",
    "de",
    "es",
    "ru",
)


@dataclass(slots=True)
class TranslationConfig:
    """Serializable configuration for translation initialization and execution."""

    source_language: str = "ja"
    target_language: str = "en"
    translator: str = "Google"
    style: str = "Default"
    custom_prompt: str = ""
    batch_size_pages: int = 3
    use_context_memory: bool = False
    local_llm_server_url: str = "http://127.0.0.1:8080"
    local_llm_model: str = "gpt-4o"
    gemini_api_key: str = ""
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-v4-flash"
    deepseek_thinking: bool = False

    @classmethod
    def from_value(cls, value: "TranslationConfig | dict[str, Any] | None") -> "TranslationConfig":
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            return cls()

        return cls(
            source_language=str(value.get("source_language", "ja") or "ja"),
            target_language=str(value.get("target_language", "en") or "en"),
            translator=str(value.get("translator", "Google") or "Google"),
            style=str(value.get("style", "Default") or "Default"),
            custom_prompt=str(value.get("custom_prompt", "") or ""),
            batch_size_pages=_coerce_positive_int(value.get("batch_size_pages"), 3),
            use_context_memory=bool(value.get("use_context_memory", False)),
            local_llm_server_url=str(value.get("local_llm_server_url", "http://127.0.0.1:8080") or "http://127.0.0.1:8080"),
            local_llm_model=str(value.get("local_llm_model", "gpt-4o") or "gpt-4o"),
            gemini_api_key=str(value.get("gemini_api_key", "") or ""),
            deepseek_api_key=str(value.get("deepseek_api_key", "") or ""),
            deepseek_model=str(value.get("deepseek_model", "deepseek-v4-flash") or "deepseek-v4-flash"),
            deepseek_thinking=bool(value.get("deepseek_thinking", False)),
        )

    @property
    def translator_key(self) -> str:
        return normalize_translator_name(self.translator)

    @property
    def supports_page_batch(self) -> bool:
        return self.translator_key in {"gemini", "local_llm", "deepseek"}

    def effective_prompt(self) -> str:
        style_prompt = STYLE_PROMPTS.get(self.style, "")
        custom_prompt = str(self.custom_prompt or "").strip()

        if self.style == "Custom":
            return custom_prompt
        if style_prompt and custom_prompt:
            return f"{style_prompt}\n\nAdditional instructions: {custom_prompt}"
        if style_prompt:
            return style_prompt
        return custom_prompt

    def to_metadata(self) -> dict[str, Any]:
        return {
            "source_language": self.source_language,
            "target_language": self.target_language,
            "translator": self.translator,
            "style": self.style,
            "custom_prompt": self.effective_prompt(),
            "batch_size_pages": int(self.batch_size_pages),
            "use_context_memory": bool(self.use_context_memory),
            "local_llm_server_url": self.local_llm_server_url,
            "local_llm_model": self.local_llm_model,
            "gemini_api_key": self.gemini_api_key,
            "deepseek_api_key": self.deepseek_api_key,
            "deepseek_model": self.deepseek_model,
            "deepseek_thinking": bool(self.deepseek_thinking),
        }


def normalize_translator_name(name: str) -> str:
    cleaned = str(name or "").strip().lower()
    mapping = {
        "gemini": "gemini",
        "local llm": "local_llm",
        "local_llm": "local_llm",
        "deepseek": "deepseek",
        "google": "google",
        "nllb": "nllb",
        "baidu": "baidu",
        "bing": "bing",
    }
    return mapping.get(cleaned, "google")


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        return int(default)
    return parsed if parsed > 0 else int(default)


__all__ = [
    "LANGUAGE_CHOICES",
    "STYLE_PROMPTS",
    "TRANSLATOR_CHOICES",
    "TranslationConfig",
    "normalize_translator_name",
]
