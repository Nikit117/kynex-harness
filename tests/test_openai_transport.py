"""HTTP contracts for Azure and standard OpenAI-compatible model endpoints."""

from __future__ import annotations

import io
import json
import os
import socket
import urllib.request

import httpx
import langchain_openai
import pytest
from langchain_core.messages import HumanMessage

from autonomous_document_harness.core.config import RuntimeConfig
from autonomous_document_harness.deep.model import build_tracked_chat_model, ensure_azure_env
from autonomous_document_harness.deep.usage import thread_reset, thread_snapshot
from autonomous_document_harness.models.gateway import ModelGateway
from autonomous_document_harness.models.routing import should_use_model
from autonomous_document_harness.models.schemas import ModelRequest
from autonomous_document_harness.tools.vision_llm import vision_configured


@pytest.fixture
def model_env(monkeypatch):
    for key in tuple(os.environ):
        if key.startswith(("ADH_", "AZURE_OPENAI_", "OPENAI_")):
            monkeypatch.delenv(key)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("ADH_MODEL_API_KEY", "fixture-key")
    monkeypatch.setenv("ADH_MODEL_DEPLOYMENT", "fixture-model")
    monkeypatch.setenv("ADH_MODEL_API_VERSION", "2024-10-21")

    def no_network(*args, **kwargs):
        raise AssertionError("Unexpected network access in a transport test")

    monkeypatch.setattr(socket, "getaddrinfo", no_network)


def _response(message, finish_reason="stop"):
    return {
        "id": "chatcmpl-fixture",
        "object": "chat.completion",
        "created": 0,
        "model": "fixture-model",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    }


@pytest.mark.parametrize("backend", ["openai", "openai-compatible", "default"])
def test_agent_transport_preserves_tool_calls_and_token_usage(backend, model_env, monkeypatch):
    """Catch Azure URLs on standard endpoints and changes to the default Azure path."""
    base = "https://models.example/v1" if backend == "openai" else "https://models.example"
    monkeypatch.setenv("ADH_MODEL_BASE_URL", base)
    if backend != "default":
        monkeypatch.setenv("ADH_MODEL_BACKEND", backend)
    ensure_azure_env()
    requests = []

    def send(request):
        requests.append(request)
        return httpx.Response(
            200,
            request=request,
            json=_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-fixture",
                            "type": "function",
                            "function": {"name": "read_page", "arguments": '{"page_number":1}'},
                        }
                    ],
                },
                finish_reason="tool_calls",
            ),
        )

    client = httpx.Client(transport=httpx.MockTransport(send))
    for name in ("ChatOpenAI", "AzureChatOpenAI"):
        constructor = getattr(langchain_openai, name)

        def with_client(*args, _constructor=constructor, **kwargs):
            return _constructor(*args, http_client=client, **kwargs)

        monkeypatch.setattr(langchain_openai, name, with_client)
    model = build_tracked_chat_model(max_tokens=17, request_timeout=9, max_retries=0)
    thread_reset()
    response = model.bind_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "read_page",
                    "description": "Read a document page.",
                    "parameters": {
                        "type": "object",
                        "properties": {"page_number": {"type": "integer"}},
                    },
                },
            }
        ]
    ).invoke([HumanMessage(content="Read page one.")])
    assert response.tool_calls[0]["args"] == {"page_number": 1}
    assert thread_snapshot()["prompt_tokens"] == 7
    assert thread_snapshot()["completion_tokens"] == 3
    assert len(requests) == 1
    request = requests[0]
    payload = json.loads(request.content)
    assert payload["tools"][0]["function"]["name"] == "read_page"
    assert payload.get("max_completion_tokens", payload.get("max_tokens")) == 17
    if backend == "openai":
        assert str(request.url) == base + "/chat/completions"
        assert request.headers["authorization"] == "Bearer fixture-key"
        assert "api-key" not in request.headers
        assert payload["model"] == "fixture-model"
    else:
        assert str(request.url) == (
            base + "/openai/deployments/fixture-model/chat/completions?api-version=2024-10-21"
        )
        assert request.headers["api-key"] == "fixture-key"


@pytest.mark.parametrize("backend", ["openai", "openai-compatible"])
def test_vision_transport_preserves_images_json_mode_and_usage(backend, model_env, monkeypatch):
    """Catch vision being disabled or still sent to Azure after selecting standard HTTP."""
    base = "https://models.example/v1" if backend == "openai" else "https://models.example"
    monkeypatch.setenv("ADH_MODEL_BACKEND", backend)
    monkeypatch.setenv("ADH_MODEL_BASE_URL", base)
    ensure_azure_env()
    config = RuntimeConfig.from_env()
    gateway = ModelGateway(config)
    assert gateway.model_reasoning_available()
    assert should_use_model(config, "vision")
    assert vision_configured(config)
    requests = []

    def urlopen(request, **kwargs):
        requests.append(request)
        return io.BytesIO(json.dumps(_response({"role": "assistant", "content": "{}"})).encode())

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    image = "data:image/png;base64,fixture"
    response = gateway.complete(
        ModelRequest(
            kind="vision",
            prompt="Read the image.",
            images=[image],
            max_tokens=17,
            response_format="json_object",
        )
    )
    assert response.text == "{}"
    assert response.tokens_in == 7 and response.tokens_out == 3
    assert len(requests) == 1
    request = requests[0]
    payload = json.loads(request.data)
    assert payload["messages"][0]["content"] == [
        {"type": "text", "text": "Read the image."},
        {"type": "image_url", "image_url": {"url": image}},
    ]
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["max_completion_tokens"] == 17
    headers = {key.lower(): value for key, value in request.header_items()}
    if backend == "openai":
        assert request.full_url == base + "/chat/completions"
        assert headers["authorization"] == "Bearer fixture-key"
        assert "api-key" not in headers
        assert payload["model"] == "fixture-model"
    else:
        assert request.full_url == (
            base + "/openai/deployments/fixture-model/chat/completions?api-version=2024-10-21"
        )
        assert headers["api-key"] == "fixture-key"
        assert "model" not in payload
