"""Hand-written OpenAI-compatible chat-completions client.

Deliberately plain `httpx` rather than the `openai` SDK: the agent must run
against *any* endpoint that speaks the /v1/chat/completions dialect, and the
retry, usage and malformed-tool-call behaviour the agent depends on is all
easier to guarantee when the wire format is explicit.

The service contract is that a chat turn never 500s because of a weird model
response, so every parse step degrades instead of raising; the only exception
this module raises on purpose is `LLMError` for a non-retryable HTTP failure.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..config import Config

__all__ = [
    "LLMError",
    "ToolCallRequest",
    "Usage",
    "LLMResponse",
    "LLMClient",
]

logger = logging.getLogger(__name__)

#: The literal `load_config` stores when no key is configured. Sending it as a
#: bearer token is noise, and a few hosted gateways treat a bogus token as a 401.
_PLACEHOLDER_KEYS = frozenset({"ollama", "none", "null", "changeme", "your-api-key"})

#: 429 and 5xx are transient; every other 4xx is a bug in our request and will
#: fail identically on every retry, so it is surfaced immediately.
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})

#: Roughly 3.6 characters per token. Deliberately not 4: Devanagari and
#: Romanised Hindi tokenize denser than English, so 4 over-counts the budget.
_CHARS_PER_TOKEN = 3.6

_BODY_PREVIEW_CHARS = 400


class LLMError(RuntimeError):
    """A model call that cannot be completed after the configured retries."""


@dataclass
class ToolCallRequest:
    id: str
    name: str
    #: Already JSON-parsed. `{}` when the model emitted malformed JSON, so a
    #: tool call with junk arguments reaches the registry's validator as a
    #: normal validation error rather than crashing the turn.
    arguments: dict[str, Any]


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    #: "endpoint" when the provider reported real counts, "estimated" when we
    #: counted characters. The eval report must not present estimates as facts.
    source: str = "endpoint"

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __iadd__(self, other: "Usage") -> "Usage":
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            source="endpoint" if "endpoint" in (self.source, other.source) else "estimated",
        )


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCallRequest]
    usage: Usage
    finish_reason: str
    raw: dict[str, Any]
    #: Non-fatal complaints about the response shape, e.g. "malformed JSON in
    #: tool_calls[0].function.arguments". Surfaced for diagnostics; the caller
    #: gets a usable response regardless.
    parse_problems: list[str] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


def _truncate(text: str, limit: int = _BODY_PREVIEW_CHARS) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= limit else f"{text[:limit]}... [truncated]"


def _estimate_tokens(text: str) -> int:
    return int(len(text) / _CHARS_PER_TOKEN) if text else 0


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _parse_arguments(raw: Any) -> tuple[dict[str, Any], str | None]:
    """Coerce a provider's `function.arguments` into a dict, never raising."""
    if raw is None:
        return {}, None
    if isinstance(raw, dict):
        return dict(raw), None
    if not isinstance(raw, str):
        return {}, f"arguments of type {type(raw).__name__} are not usable"
    text = raw.strip()
    if not text:
        return {}, None
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError) as exc:
        return {}, f"malformed JSON in function.arguments ({exc.__class__.__name__})"
    if isinstance(parsed, dict):
        return parsed, None
    return {}, f"function.arguments decoded to {type(parsed).__name__}, not an object"


def _as_positive_int(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return number if number > 0 else 0


class LLMClient:
    """Blocking chat-completions caller with retries and defensive parsing."""

    def __init__(
        self,
        config: Config,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout_s: float | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._config = config
        self._url = config.chat_completions_url
        timeout = timeout_s if timeout_s is not None else config.llm.request_timeout_s
        # An injected client (tests, or a caller with its own pool) wins; the
        # transport-only path exists so tests can pass MockTransport without
        # monkeypatching module globals.
        self._client = client or httpx.Client(transport=transport, timeout=timeout)

    # -- introspection -----------------------------------------------------

    @property
    def url(self) -> str:
        return self._url

    @property
    def model(self) -> str:
        return self._config.llm.model

    # -- request shaping ---------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = (self._config.llm.api_key or "").strip()
        # Local Ollama ignores the header but hosted providers (OpenAI, Groq,
        # Together, ...) reject the request without it, so the portable rule is
        # to send a real key and omit a placeholder one.
        if key and key.casefold() not in _PLACEHOLDER_KEYS:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _base_payload(self, messages: list[dict], tools: list[dict] | None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._config.llm.model,
            "messages": list(messages),
            "temperature": self._config.llm.temperature,
            "stream": False,
            # Ollama reads its context size from `options`; gateways that do not
            # know the key ignore unknown body fields, so it stays harmless.
            "options": {"num_ctx": self._config.llm.num_ctx},
        }
        if tools:
            payload["tools"] = tools
        return payload

    def build_payload(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> dict[str, Any]:
        """The exact JSON body that would be posted. Public so tests can assert it."""
        payload = self._base_payload(messages, tools)
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        return payload

    # -- transport ---------------------------------------------------------

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        attempts = max(1, int(self._config.llm.max_retries) + 1)
        backoff = max(0.0, float(self._config.llm.retry_backoff_s))
        last_error: str = "no attempt made"

        for attempt in range(attempts):
            retryable: str | None = None
            try:
                response = self._client.post(
                    self._url, json=payload, headers=self._headers()
                )
            except httpx.TransportError as exc:
                # Covers ConnectError and TimeoutException, which subclass it.
                retryable = f"{exc.__class__.__name__}: {exc}"
            else:
                status = response.status_code
                if status in _RETRYABLE_STATUS or status >= 500:
                    retryable = f"HTTP {status}: {_truncate(response.text)}"
                elif status >= 400:
                    raise LLMError(
                        f"LLM request to {self._url} failed with HTTP {status}: "
                        f"{_truncate(response.text)}"
                    )
                else:
                    return self._decode(response)

            last_error = retryable or "unknown error"
            if attempt < attempts - 1:
                logger.warning(
                    "llm call failed (attempt %d/%d): %s", attempt + 1, attempts, last_error
                )
                time.sleep(backoff * (2**attempt))

        raise LLMError(
            f"LLM request to {self._url} failed after {attempts} attempt(s): {last_error}"
        )

    def _decode(self, response: httpx.Response) -> dict[str, Any]:
        try:
            body = response.json()
        except ValueError as exc:
            raise LLMError(
                f"LLM response from {self._url} was not JSON: {_truncate(response.text)}"
            ) from exc
        if not isinstance(body, dict):
            return {"_non_object_body": body}
        return body

    # -- response parsing --------------------------------------------------

    def _parse(
        self, body: dict[str, Any], prompt_text: str
    ) -> tuple[str, list[ToolCallRequest], Usage, str, list[str]]:
        problems: list[str] = []
        raw_choices = body.get("choices")
        if not isinstance(raw_choices, list) or not raw_choices:
            problems.append("response carried no choices")
            choice: dict[str, Any] = {}
        elif isinstance(raw_choices[0], dict):
            choice = raw_choices[0]
        else:
            problems.append("choices[0] was not an object")
            choice = {}

        message = _as_dict(choice.get("message"))
        content = message.get("content")
        if content is None:
            if not message.get("tool_calls"):
                problems.append("message.content was null and there were no tool calls")
        if not isinstance(content, str):
            if content is not None:
                problems.append(f"message.content was {type(content).__name__}, not a string")
            content = ""

        tool_calls = self._parse_tool_calls(message, problems)

        finish_reason = choice.get("finish_reason")
        if not isinstance(finish_reason, str) or not finish_reason:
            finish_reason = "tool_calls" if tool_calls else "stop"

        usage = self._parse_usage(body, prompt_text, content)
        return content, tool_calls, usage, finish_reason, problems

    def _parse_tool_calls(
        self, message: dict[str, Any], problems: list[str]
    ) -> list[ToolCallRequest]:
        raw_calls = message.get("tool_calls")
        if not isinstance(raw_calls, list):
            legacy = message.get("function_call")
            if isinstance(legacy, dict):
                problems.append("used the legacy function_call field")
                raw_calls = [{"function": legacy}]
            else:
                return []

        calls: list[ToolCallRequest] = []
        for index, raw_call in enumerate(raw_calls):
            if not isinstance(raw_call, dict):
                problems.append(f"tool_calls[{index}] was not an object")
                continue
            function = _as_dict(raw_call.get("function"))
            name = function.get("name")
            if not isinstance(name, str) or not name:
                problems.append(f"tool_calls[{index}] had no function name")
                continue
            arguments, problem = _parse_arguments(function.get("arguments"))
            if problem:
                problems.append(f"tool_calls[{index}] ({name}): {problem}")
            call_id = raw_call.get("id")
            calls.append(
                ToolCallRequest(
                    id=call_id if isinstance(call_id, str) and call_id else f"call_{index}",
                    name=name,
                    arguments=arguments,
                )
            )
        return calls

    def _parse_usage(self, body: dict[str, Any], prompt_text: str, content: str) -> Usage:
        raw = _as_dict(body.get("usage"))
        prompt_tokens = _as_positive_int(
            raw.get("prompt_tokens", raw.get("input_tokens"))
        )
        completion_tokens = _as_positive_int(
            raw.get("completion_tokens", raw.get("output_tokens"))
        )
        if prompt_tokens or completion_tokens:
            return Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                source="endpoint",
            )
        return Usage(
            prompt_tokens=_estimate_tokens(prompt_text),
            completion_tokens=_estimate_tokens(content),
            source="estimated",
        )

    # -- public API --------------------------------------------------------

    def complete(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> LLMResponse:
        payload = self.build_payload(messages, tools=tools, tool_choice=tool_choice)
        body = self._post(payload)

        prompt_text = json.dumps(
            {"messages": payload["messages"], "tools": payload.get("tools")},
            ensure_ascii=False,
        )
        content, tool_calls, usage, finish_reason, problems = self._parse(body, prompt_text)
        if problems:
            logger.warning("llm response needed recovery: %s", "; ".join(problems))

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=finish_reason,
            raw=body,
            parse_problems=problems,
        )

    def health(self) -> bool:
        """True when the endpoint answers a one-token request. Used by GET /health."""
        payload = {
            "model": self._config.llm.model,
            "messages": [{"role": "user", "content": "ping"}],
            "temperature": 0.0,
            "stream": False,
            "max_tokens": 1,
            "options": {"num_ctx": self._config.llm.num_ctx},
        }
        try:
            self._post(payload)
        except (LLMError, httpx.HTTPError) as exc:
            logger.warning("llm health check failed: %s", exc)
            return False
        return True

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
