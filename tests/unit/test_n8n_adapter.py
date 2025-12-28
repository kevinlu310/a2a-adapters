"""
Unit tests for N8nAgentAdapter.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from a2a_adapter.integrations.n8n import N8nAgentAdapter
from a2a.types import Message, MessageSendParams, TextPart, Role, Part


def make_message_send_params(text: str, context_id: str | None = None) -> MessageSendParams:
    """Helper to create MessageSendParams with correct A2A types."""
    return MessageSendParams(
        message=Message(
            message_id="test-msg-id",
            role=Role.user,
            parts=[Part(root=TextPart(text=text))],
            context_id=context_id,
        )
    )


class TestN8nAdapterContextId:
    """Tests for context_id handling in N8nAgentAdapter."""

    @pytest.mark.asyncio
    async def test_to_framework_extracts_context_id(self):
        """Test that to_framework extracts context_id from params.message."""
        adapter = N8nAgentAdapter(webhook_url="http://example.com/webhook")
        
        params = make_message_send_params("hello", context_id="ctx-123")
        payload = await adapter.to_framework(params)
        
        assert payload["metadata"]["context_id"] == "ctx-123"

    @pytest.mark.asyncio
    async def test_to_framework_handles_missing_context_id(self):
        """Test that to_framework handles missing context_id gracefully."""
        adapter = N8nAgentAdapter(webhook_url="http://example.com/webhook")
        
        params = make_message_send_params("hello", context_id=None)
        payload = await adapter.to_framework(params)
        
        assert payload["metadata"]["context_id"] is None

    @pytest.mark.asyncio
    async def test_to_framework_with_template_adds_context_id_at_root(self):
        """Test that context_id is added at root level when using payload_template."""
        adapter = N8nAgentAdapter(
            webhook_url="http://example.com/webhook",
            payload_template={"custom_field": "value"}
        )
        
        params = make_message_send_params("hello", context_id="ctx-456")
        payload = await adapter.to_framework(params)
        
        # With template, context_id should be at root, not in metadata
        assert payload["context_id"] == "ctx-456"
        assert "metadata" not in payload
        assert payload["custom_field"] == "value"

    @pytest.mark.asyncio
    async def test_to_framework_with_template_respects_existing_context_id(self):
        """Test that template's context_id is not overwritten."""
        adapter = N8nAgentAdapter(
            webhook_url="http://example.com/webhook",
            payload_template={"context_id": "template-ctx"}
        )
        
        params = make_message_send_params("hello", context_id="request-ctx")
        payload = await adapter.to_framework(params)
        
        # Template's context_id should be preserved
        assert payload["context_id"] == "template-ctx"

    @pytest.mark.asyncio
    async def test_from_framework_preserves_context_id(self):
        """Test that from_framework preserves context_id on the response Message."""
        adapter = N8nAgentAdapter(webhook_url="http://example.com/webhook")
        
        params = make_message_send_params("hello", context_id="ctx-789")
        framework_output = {"output": "response text"}
        
        result = await adapter.from_framework(framework_output, params)
        
        assert isinstance(result, Message)
        assert result.context_id == "ctx-789"
        assert result.role == Role.agent

    @pytest.mark.asyncio
    async def test_from_framework_handles_missing_context_id(self):
        """Test that from_framework handles missing context_id gracefully."""
        adapter = N8nAgentAdapter(webhook_url="http://example.com/webhook")
        
        params = make_message_send_params("hello", context_id=None)
        framework_output = {"output": "response text"}
        
        result = await adapter.from_framework(framework_output, params)
        
        assert isinstance(result, Message)
        assert result.context_id is None

    @pytest.mark.asyncio
    async def test_handle_preserves_context_id_end_to_end(self):
        """Test that handle() preserves context_id through the full request/response cycle."""
        adapter = N8nAgentAdapter(webhook_url="http://example.com/webhook")
        
        # Mock the HTTP client
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"output": "n8n response"}
        mock_response.raise_for_status = MagicMock()
        
        with patch.object(adapter, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client
            
            params = make_message_send_params("test message", context_id="e2e-ctx-id")
            result = await adapter.handle(params)
            
            # Verify context_id is preserved in response
            assert result.context_id == "e2e-ctx-id"
            
            # Verify context_id was sent in the payload
            call_args = mock_client.post.call_args
            sent_payload = call_args.kwargs["json"]
            assert sent_payload["metadata"]["context_id"] == "e2e-ctx-id"


class TestN8nAdapterBasic:
    """Basic functionality tests for N8nAgentAdapter."""

    @pytest.mark.asyncio
    async def test_to_framework_extracts_message_text(self):
        """Test that to_framework extracts message text correctly."""
        adapter = N8nAgentAdapter(webhook_url="http://example.com/webhook")
        
        params = make_message_send_params("hello world")
        payload = await adapter.to_framework(params)
        
        assert payload["message"] == "hello world"

    @pytest.mark.asyncio
    async def test_to_framework_custom_message_field(self):
        """Test that custom message_field is used."""
        adapter = N8nAgentAdapter(
            webhook_url="http://example.com/webhook",
            message_field="query"
        )
        
        params = make_message_send_params("test query")
        payload = await adapter.to_framework(params)
        
        assert payload["query"] == "test query"
        assert "message" not in payload

    @pytest.mark.asyncio
    async def test_from_framework_extracts_output_field(self):
        """Test that from_framework extracts the output field."""
        adapter = N8nAgentAdapter(webhook_url="http://example.com/webhook")
        
        params = make_message_send_params("hello")
        framework_output = {"output": "the response"}
        
        result = await adapter.from_framework(framework_output, params)
        
        assert result.parts[0].root.text == "the response"

    @pytest.mark.asyncio
    async def test_from_framework_handles_array_response(self):
        """Test that from_framework handles array responses."""
        adapter = N8nAgentAdapter(webhook_url="http://example.com/webhook")
        
        params = make_message_send_params("hello")
        framework_output = [{"output": "first"}, {"output": "second"}]
        
        result = await adapter.from_framework(framework_output, params)
        
        assert "first" in result.parts[0].root.text
        assert "second" in result.parts[0].root.text

    def test_supports_streaming_returns_false(self):
        """Test that N8N adapter does not support streaming."""
        adapter = N8nAgentAdapter(webhook_url="http://example.com/webhook")
        assert adapter.supports_streaming() is False
