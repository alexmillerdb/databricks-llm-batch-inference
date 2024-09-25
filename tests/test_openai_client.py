import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from llm_batch_inference.api.openai_client import OpenAIClient
from llm_batch_inference.config.inference_config import InferenceConfig

@pytest.mark.asyncio
async def test_async_predict_chat():
    config = InferenceConfig(llm_task="chat", endpoint="test-endpoint", prompt="Hello", request_params={})
    client = OpenAIClient(config, API_ROOT="http://api.test", API_TOKEN="test-token")
    client._async_chat_completion = AsyncMock(return_value=("response", 10))

    result = await client.async_predict("test text")
    assert result == ("response", 10)
    client._async_chat_completion.assert_awaited_once_with("test text")

@pytest.mark.asyncio
async def test_async_predict_completion():
    config = InferenceConfig(llm_task="completion", endpoint="test-endpoint", prompt="Hello", request_params={})
    client = OpenAIClient(config, API_ROOT="http://api.test", API_TOKEN="test-token")
    client._async_completion = AsyncMock(return_value=("response", 10))

    result = await client.async_predict("test text")
    assert result == ("response", 10)
    client._async_completion.assert_awaited_once_with("test text")

@pytest.mark.asyncio
async def test_async_predict_embedding():
    config = InferenceConfig(llm_task="embedding", endpoint="test-endpoint", prompt=None, request_params={})
    client = OpenAIClient(config, API_ROOT="http://api.test", API_TOKEN="test-token")
    client._async_embedding = AsyncMock(return_value=([0.1, 0.2, 0.3], 10))

    result = await client.async_predict("test text")
    assert result == ([0.1, 0.2, 0.3], 10)
    client._async_embedding.assert_awaited_once_with("test text")

# @pytest.mark.asyncio
# async def test_async_predict_unsupported_task():
#     config = InferenceConfig(llm_task="unsupported", endpoint="test-endpoint", prompt=None, request_params={})
#     client = OpenAIClient(config, API_ROOT="http://api.test", API_TOKEN="test-token")

#     with pytest.raises(ValueError, match="Unsupported llm_task: unsupported"):
#         await client.async_predict("test text")