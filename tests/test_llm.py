from __future__ import annotations

import httpx
import pytest

from aircraft_damage.llm import OllamaClient, OllamaError, strip_thinking

HOST = "http://ollama.test:11434"


def _client(handler: object) -> OllamaClient:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    return OllamaClient(HOST, "qwen3:4b", client=httpx.Client(transport=transport))


def test_strip_thinking_removes_a_reasoning_block() -> None:
    assert strip_thinking("<think>weighing options</think>\n\n## Finding\nA dent.") == (
        "## Finding\nA dent."
    )


def test_strip_thinking_handles_multiline_blocks() -> None:
    text = "<think>\nline one\nline two\n</think>Result"

    assert strip_thinking(text) == "Result"


def test_strip_thinking_leaves_ordinary_text_alone() -> None:
    assert strip_thinking("## Finding\nA crack.") == "## Finding\nA crack."


def test_chat_posts_the_configured_model_and_messages() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["json"] = httpx.Request("POST", "http://x", content=request.content).content
        import json

        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "A report."}})

    result = _client(handler).chat("You are an inspector.", "Evidence here.")

    payload = seen["payload"]
    assert seen["url"] == f"{HOST}/api/chat"
    assert payload["model"] == "qwen3:4b"  # type: ignore[index]
    assert payload["stream"] is False  # type: ignore[index]
    assert payload["messages"] == [  # type: ignore[index]
        {"role": "system", "content": "You are an inspector."},
        {"role": "user", "content": "Evidence here."},
    ]
    assert result == "A report."


def test_chat_strips_inline_thinking_from_the_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": "<think>hmm</think>Done."}})

    assert _client(handler).chat("sys", "user") == "Done."


def test_chat_retries_without_the_think_field_when_the_server_rejects_it() -> None:
    attempts: list[bool] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        payload = json.loads(request.content)
        attempts.append("think" in payload)
        if "think" in payload:
            return httpx.Response(400, text="unknown field think")
        return httpx.Response(200, json={"message": {"content": "Done."}})

    assert _client(handler).chat("sys", "user") == "Done."
    assert attempts == [True, False]


def test_chat_raises_on_a_server_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="model runner crashed")

    with pytest.raises(OllamaError, match="500"):
        _client(handler).chat("sys", "user")


def test_chat_raises_on_an_empty_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": "   "}})

    with pytest.raises(OllamaError, match="empty"):
        _client(handler).chat("sys", "user")


def test_chat_raises_when_the_host_is_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(OllamaError, match="Could not reach Ollama"):
        _client(handler).chat("sys", "user")


def test_is_available_is_true_when_tags_responds() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": []})

    assert _client(handler).is_available() is True


def test_is_available_is_false_when_the_host_refuses() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    assert _client(handler).is_available() is False


def test_has_model_matches_the_configured_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "qwen3:4b"}, {"name": "llama3:8b"}]})

    client = _client(handler)
    assert client.available_models() == ["qwen3:4b", "llama3:8b"]
    assert client.has_model() is True
