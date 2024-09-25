import pytest
from pydantic import ValidationError
from llm_batch_inference.config.inference_config import InferenceConfig

def test_default_values():
    config = InferenceConfig()
    assert config.endpoint == "databricks-meta-llama-3-1-70b-instruct"
    assert config.timeout == 300
    assert config.max_retries_backpressure == 3
    assert config.max_retries_other == 3
    assert config.prompt is None
    assert config.request_params == {}
    assert config.concurrency == 15
    assert config.logging_interval == 40
    assert config.enable_logging is True
    assert config.llm_task == ""
    assert config.model_config == {"extra": "forbid"}

def test_custom_values():
    config = InferenceConfig(
        endpoint="custom-endpoint",
        timeout=100,
        max_retries_backpressure=5,
        max_retries_other=5,
        prompt="Test prompt",
        request_params={"param1": "value1"},
        concurrency=10,
        logging_interval=20,
        enable_logging=False,
        llm_task="chat"
    )
    assert config.endpoint == "custom-endpoint"
    assert config.timeout == 100
    assert config.max_retries_backpressure == 5
    assert config.max_retries_other == 5
    assert config.prompt == "Test prompt"
    assert config.request_params == {"param1": "value1"}
    assert config.concurrency == 10
    assert config.logging_interval == 20
    assert config.enable_logging is False
    assert config.llm_task == "chat"
    assert config.model_config == {"extra": "forbid"}

def test_invalid_llm_task():
    with pytest.raises(ValidationError):
        InferenceConfig(llm_task="invalid_task")

def test_invalid_timeout():
    with pytest.raises(ValidationError):
        InferenceConfig(timeout=-1)

def test_invalid_concurrency():
    with pytest.raises(ValidationError):
        InferenceConfig(concurrency=-1)