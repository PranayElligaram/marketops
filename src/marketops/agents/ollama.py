"""Minimal local Ollama chat client (httpx). Only loopback / local-container hosts are accepted."""

from __future__ import annotations

from typing import Any, Protocol
from urllib.parse import urlparse

import httpx

ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1", "host.docker.internal", "ollama"}


class LLMError(Exception):
    """Base class; `status` is a short machine-readable reason."""

    status = "error"


class OllamaUnavailable(LLMError):
    status = "ollama_unavailable"


class ModelUnavailable(LLMError):
    status = "model_unavailable"


class LLMTimeout(LLMError):
    status = "timeout"


class BadModelResponse(LLMError):
    status = "bad_response"


class ChatClient(Protocol):
    model: str

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        fmt: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]: ...


def validate_base_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError(
            f"Ollama host must be one of {sorted(ALLOWED_HOSTS)}; external LLM hosts are refused."
        )
    return url.rstrip("/")


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 300.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = validate_base_url(base_url)
        self.model = model
        self.timeout = timeout
        self._http = httpx.Client(transport=transport)

    def _request(self, method: str, path: str, timeout: float, **kw: Any) -> httpx.Response:
        try:
            return self._http.request(method, f"{self.base_url}{path}", timeout=timeout, **kw)
        except httpx.TimeoutException:
            raise LLMTimeout("Ollama request timed out.") from None
        except httpx.RequestError:
            raise OllamaUnavailable("Cannot reach the local Ollama service.") from None

    def installed_models(self) -> list[str]:
        r = self._request("GET", "/api/tags", 5.0)
        if r.status_code != 200:
            raise OllamaUnavailable(f"Ollama returned HTTP {r.status_code}.")
        return [str(m.get("name")) for m in r.json().get("models", [])]

    def status(self) -> dict[str, Any]:
        try:
            names = self.installed_models()
        except LLMError as exc:
            return {
                "ollama": "unavailable",
                "model": self.model,
                "model_installed": False,
                "detail": str(exc),
            }
        ok = self.model in names
        return {
            "ollama": "up",
            "model": self.model,
            "model_installed": ok,
            "detail": None if ok else f"Model {self.model} is not installed.",
        }

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        fmt: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"temperature": 0.1, "num_ctx": 8192},
        }
        if tools:
            body["tools"] = tools
            body["options"]["num_predict"] = 400  # tool selection is a few short calls
        if fmt:
            body["format"] = fmt
            body["options"]["num_predict"] = 1200  # bound a runaway generation
        r = self._request("POST", "/api/chat", timeout or self.timeout, json=body)
        if r.status_code == 404:
            raise ModelUnavailable(f"Model {self.model} is not installed in Ollama.")
        if r.status_code != 200:
            raise BadModelResponse(f"Ollama returned HTTP {r.status_code}.")
        try:
            body_out = r.json()
            message = body_out["message"]
        except (ValueError, KeyError, TypeError):
            raise BadModelResponse("Malformed Ollama response.") from None
        if not isinstance(message, dict):
            raise BadModelResponse("Malformed Ollama response.")
        counts = (body_out.get("prompt_eval_count"), body_out.get("eval_count"))
        if all(isinstance(c, int) for c in counts):
            message["_tokens"] = {"prompt": counts[0], "output": counts[1]}
        return message
