import pytest
import asyncio
from unittest.mock import AsyncMock, Mock
from llm_batch_inference.inference.batch_processor import BatchProcessor

@pytest.fixture
def mock_engine():
    engine = Mock()
    engine.async_infer = AsyncMock()
    engine.infer = Mock()
    return engine

@pytest.fixture
def mock_config():
    config = Mock()
    config.concurrency = 2
    config.llm_task = "completion"  # Add this line
    config.logging_interval = 40
    config.request_params = {"temperature": 0, "max_tokens": 500}
    config.endpoint = "databricks-meta-llama-3-1-70b-instruct"
    config.timeout = 300
    config.max_retries_backpressure = 3
    config.max_retries_other = 3
    config.prompt = ""
    return config

@pytest.fixture
def batch_processor(mock_engine, mock_config):
    return BatchProcessor(mock_engine, mock_config)

@pytest.mark.asyncio
async def test_process_item_async_success(batch_processor, mock_engine):
    item = (1, "test text")
    mock_engine.async_infer.return_value = ("processed content", 5)

    result = await batch_processor.process_item_async(item)

    assert result == (1, "processed content", 5, None)
    mock_engine.async_infer.assert_called_once_with("test text")

@pytest.mark.asyncio
async def test_process_item_async_failure(batch_processor, mock_engine):
    item = (1, "test text")
    mock_engine.async_infer.side_effect = Exception("Inference error")

    result = await batch_processor.process_item_async(item)

    assert result == (1, None, 0, "Inference error")
    mock_engine.async_infer.assert_called_once_with("test text")

def test_process_item_success(batch_processor, mock_engine):
    item = (1, "test text")
    mock_engine.infer.return_value = ("processed content", 5)

    result = batch_processor.process_item(item)

    assert result == (1, "processed content", 5, None)
    mock_engine.infer.assert_called_once_with("test text")

def test_process_item_failure(batch_processor, mock_engine):
    item = (1, "test text")
    mock_engine.infer.side_effect = Exception("Inference error")

    result = batch_processor.process_item(item)

    assert result == (1, None, 0, "Inference error")
    mock_engine.infer.assert_called_once_with("test text")

@pytest.mark.asyncio
async def test_process_batch_async(batch_processor, mock_engine):
    items = [(1, "text1"), (2, "text2")]
    mock_engine.async_infer.side_effect = [("content1", 5), ("content2", 6)]

    results = await batch_processor.process_batch_async(items)

    assert results == [
        (1, "content1", 5, None),
        (2, "content2", 6, None)
    ]
    mock_engine.async_infer.assert_any_call("text1")
    mock_engine.async_infer.assert_any_call("text2")

def test_process_batch(batch_processor, mock_engine):
    items = [(1, "text1"), (2, "text2")]
    mock_engine.infer.side_effect = [("content1", 5), ("content2", 6)]

    results = batch_processor.process_batch(items)

    assert results == [
        (1, "content1", 5, None),
        (2, "content2", 6, None)
    ]
    mock_engine.infer.assert_any_call("text1")
    mock_engine.infer.assert_any_call("text2")