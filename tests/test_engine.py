import pytest
from unittest.mock import AsyncMock, Mock
from llm_batch_inference.inference.engine import InferenceEngine
from llm_batch_inference.api.openai_client import OpenAIClient

@pytest.fixture
def mock_client():
    client = Mock(spec=OpenAIClient)
    client.config = Mock()
    return client

def test_infer(mock_client):
    mock_client.predict.return_value = ("generated_text", 100)
    engine = InferenceEngine(client=mock_client)
    
    result = engine.infer("test text")
    
    assert result == ("generated_text", 100)
    mock_client.predict.assert_called_once_with("test text")

@pytest.mark.asyncio
async def test_async_infer(mock_client):
    mock_client.async_predict = AsyncMock(return_value=("generated_text", 100))
    engine = InferenceEngine(client=mock_client)
    
    result = await engine.async_infer("test text")
    
    assert result == ("generated_text", 100)
    mock_client.async_predict.assert_called_once_with("test text")

def test_get_task_type(mock_client):
    mock_client.config.llm_task = "completion"
    engine = InferenceEngine(client=mock_client)
    
    result = engine.get_task_type()
    
    assert result == "completion"

def test_is_embedding_task_true(mock_client):
    mock_client.config.llm_task = "embedding"
    engine = InferenceEngine(client=mock_client)
    
    result = engine.is_embedding_task()
    
    assert result is True

def test_is_embedding_task_false(mock_client):
    mock_client.config.llm_task = "completion"
    engine = InferenceEngine(client=mock_client)
    
    result = engine.is_embedding_task()
    
    assert result is False