"""HTTP client for a local Ollama server.

The model this talks to (``qwen3:4b``) has no vision encoder. Nothing in this
module accepts or transmits image data — only text.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)

CHAT_ENDPOINT = "/api/chat"
TAGS_ENDPOINT = "/api/tags"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TIMEOUT_S = 600
PROBE_TIMEOUT_S = 3.0
KEEP_ALIVE = "5m"
HTTP_OK = 200
HTTP_BAD_REQUEST = 400
THINKING_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
ORPHAN_THINKING_CLOSE_PATTERN = re.compile(r"^.*?</think>", re.DOTALL)


class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable or returns something unusable."""


def strip_thinking(text: str) -> str:
    """Remove inline reasoning, whether wrapped in <think> tags or not.

    Ollama versions that surface Qwen3's reasoning in a separate ``thinking``
    field need no help here, but older builds inline it into the content. In
    practice, the opening ``<think>`` tag lives in the model's chat template
    and is never present in the returned text — only the closing tag is —
    so a response with reasoning has an orphan ``</think>`` with nothing
    preceding it but the reasoning itself.

    Args:
        text: Raw assistant content.

    Returns:
        The content with reasoning removed and surrounding space trimmed.
    """
    if "<think>" in text:
        return THINKING_PATTERN.sub("", text).strip()
    if "</think>" in text:
        return ORPHAN_THINKING_CLOSE_PATTERN.sub("", text, count=1).strip()
    return text.strip()


class OllamaClient:
    """Talks to an Ollama server over HTTP."""

    def __init__(
        self,
        host: str,
        model: str,
        timeout_s: int = DEFAULT_TIMEOUT_S,
        client: httpx.Client | None = None,
    ) -> None:
        """Initialise the client.

        Args:
            host: Base URL of the Ollama server, e.g. ``http://localhost:11434``.
            model: Model tag to generate with, e.g. ``qwen3:4b``.
            timeout_s: Request timeout for generation calls.
            client: An ``httpx.Client`` to reuse; tests inject a mock transport.
        """
        self.host = host.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self._client = client if client is not None else httpx.Client(timeout=timeout_s)

    def is_available(self) -> bool:
        """Report whether the server answers at all.

        Returns:
            ``True`` if ``/api/tags`` responds with 200.
        """
        try:
            response = self._client.get(f"{self.host}{TAGS_ENDPOINT}", timeout=PROBE_TIMEOUT_S)
        except httpx.HTTPError:
            return False
        return response.status_code == HTTP_OK

    def available_models(self) -> list[str]:
        """List the model tags the server has pulled.

        Returns:
            Model names, or an empty list if the server cannot be reached.
        """
        try:
            response = self._client.get(f"{self.host}{TAGS_ENDPOINT}", timeout=PROBE_TIMEOUT_S)
            response.raise_for_status()
        except httpx.HTTPError:
            return []
        models: list[dict[str, Any]] = response.json().get("models", [])
        return [str(entry["name"]) for entry in models]

    def has_model(self) -> bool:
        """Report whether the configured model has been pulled.

        Returns:
            ``True`` if ``self.model`` appears in the server's tag list.
        """
        return self.model in self.available_models()

    def chat(self, system: str, user: str, *, temperature: float = DEFAULT_TEMPERATURE) -> str:
        """Send a system/user pair and return the assistant's text.

        Args:
            system: The system prompt.
            user: The user message.
            temperature: Sampling temperature.

        Returns:
            The assistant's content, with reasoning blocks stripped.

        Raises:
            OllamaError: If the server is unreachable, errors, or returns nothing.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            # Qwen3 is a hybrid reasoning model; on a CPU-only host the thinking
            # pass costs far more than it buys for this templated report.
            "think": False,
            "options": {"temperature": temperature},
            "keep_alive": KEEP_ALIVE,
        }

        response = self._post_chat(payload)
        if response.status_code == HTTP_BAD_REQUEST:
            logger.warning("Ollama rejected the 'think' field; retrying without it")
            payload.pop("think")
            response = self._post_chat(payload)

        if response.status_code != HTTP_OK:
            raise OllamaError(f"Ollama returned {response.status_code}: {response.text[:200]}")

        content = str(response.json().get("message", {}).get("content", ""))
        cleaned = strip_thinking(content)
        if not cleaned:
            raise OllamaError(f"Ollama returned an empty response from {self.model}")
        return cleaned

    def _post_chat(self, payload: dict[str, Any]) -> httpx.Response:
        """POST to the chat endpoint, translating transport errors.

        Args:
            payload: The JSON request body.

        Returns:
            The raw response, whatever its status code.

        Raises:
            OllamaError: If the request could not be delivered.
        """
        try:
            return self._client.post(
                f"{self.host}{CHAT_ENDPOINT}",
                json=payload,
                timeout=self.timeout_s,
            )
        except httpx.HTTPError as exc:
            raise OllamaError(f"Could not reach Ollama at {self.host}: {exc}") from exc
