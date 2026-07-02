from __future__ import annotations

import json
import re
import time
from typing import Any

from app.core.config import settings

_MAX_RETRIES = 2
_BASE_BACKOFF_SECONDS = 1.5
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _clean_schema_for_gemini(schema: Any) -> Any:
    if hasattr(schema, "model_json_schema"):
        schema = schema.model_json_schema()
    if isinstance(schema, dict):
        cleaned = {}
        for k, v in schema.items():
            if k == "additionalProperties":
                continue
            cleaned[k] = _clean_schema_for_gemini(v)
        return cleaned
    elif isinstance(schema, list):
        return [_clean_schema_for_gemini(item) for item in schema]
    return schema


class LLMClient:
    def __init__(self, provider: str | None = None, api_key: str | None = None, model: str | None = None):
        self.provider = (provider or settings.llm_provider).lower()
        if self.provider == "mistral":
            self.api_key = api_key or settings.mistral_api_key
            self.model = model or settings.mistral_model
        else:
            self.provider = "gemini"
            self.api_key = api_key or settings.gemini_api_key
            self.model = model or settings.gemini_model

    def is_available(self) -> bool:
        if settings.gemini_mock:
            return False
        return bool(self.api_key)

    def generate_json(self, prompt: str, temperature: float, schema: Any = None) -> dict[str, Any]:
        if not self.is_available():
            raise RuntimeError("LLM client is in mock mode or missing API key.")

        if self.provider == "mistral":
            return self._mistral_generate(prompt, temperature)
        return self._gemini_generate(prompt, temperature, schema)

    def _gemini_generate(self, prompt: str, temperature: float, schema: Any = None) -> dict[str, Any]:
        try:
            from google import genai
            from google.genai import types
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("google-genai not installed.") from exc

        client = genai.Client(api_key=self.api_key)
        last_error: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                config_args = {
                    "temperature": temperature,
                    "response_mime_type": "application/json",
                    "max_output_tokens": 16384,
                }
                if schema:
                    config_args["response_schema"] = _clean_schema_for_gemini(schema)
                
                config = types.GenerateContentConfig(**config_args)
                
                response = client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
                text = response.text or ""
                return _safe_json_loads(text)
            except Exception as exc:  # pragma: no cover - provider/runtime dependent
                last_error = exc
                if attempt >= _MAX_RETRIES or not _is_retryable_error(exc):
                    raise RuntimeError(f"Gemini request failed: {exc}") from exc
                time.sleep(_BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)))

        raise RuntimeError(f"Gemini request failed after retries: {last_error}")

    def _mistral_generate(self, prompt: str, temperature: float) -> dict[str, Any]:
        try:
            from mistralai import Mistral
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("mistralai not installed.") from exc

        client = Mistral(api_key=self.api_key)
        last_error: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                chat_response = client.chat.complete(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=temperature,
                )
                content = ""
                if chat_response and chat_response.choices:
                    content = chat_response.choices[0].message.content or ""
                return _safe_json_loads(content)
            except Exception as exc:  # pragma: no cover - provider/runtime dependent
                last_error = exc
                if attempt >= _MAX_RETRIES or not _is_retryable_error(exc):
                    raise RuntimeError(f"Mistral request failed: {exc}") from exc
                time.sleep(_BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)))

        raise RuntimeError(f"Mistral request failed after retries: {last_error}")


def _safe_json_loads(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        try:
            debug_path = "/Users/pedroguimaraes/.gemini/antigravity-ide/brain/ba728e09-c6ba-43a7-b9f5-eeae21fac0b7/scratch/failed_gemini_output.txt"
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise e
        try:
            return json.loads(match.group(0))
        except Exception:
            raise e


def _is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, json.JSONDecodeError):
        return True

    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status in _RETRYABLE_STATUS_CODES:
        return True

    code = getattr(exc, "code", None)
    if isinstance(code, int) and code in _RETRYABLE_STATUS_CODES:
        return True

    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "429",
            "500",
            "502",
            "503",
            "504",
            "unavailable",
            "resource_exhausted",
            "rate limit",
            "internal",
            "timeout",
            "temporar",
        )
    )
