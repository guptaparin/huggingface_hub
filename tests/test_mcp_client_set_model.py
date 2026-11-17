import pytest


pytest.importorskip("mcp")

from huggingface_hub.inference._mcp.mcp_client import MCPClient  # noqa: E402


def test_set_model_updates_model_and_provider_and_keeps_state():
    client = MCPClient(model="model-a", provider="together")
    client.sessions["tool"] = object()  # type: ignore[assignment]
    client.available_tools.append({"type": "function"})  # type: ignore[arg-type]

    client.set_model("model-b", provider="fireworks-ai")

    assert client.payload_model == "model-b"
    assert client.client.model == "model-b"
    assert client.client.provider == "fireworks-ai"
    assert "tool" in client.sessions
    assert len(client.available_tools) == 1


def test_set_model_keeps_provider_when_not_given():
    client = MCPClient(model="model-a", provider="together")
    client.set_model("model-b")
    assert client.client.provider == "together"


def test_set_model_with_base_url_only_changes_payload_model():
    client = MCPClient(model="model-a", base_url="http://localhost:8000/v1")
    client.set_model("model-b")
    assert client.payload_model == "model-b"
    assert client.client.model == "http://localhost:8000/v1"


def test_set_model_rejects_empty_model():
    client = MCPClient(model="model-a")
    with pytest.raises(ValueError):
        client.set_model("")
    assert client.payload_model == "model-a"
