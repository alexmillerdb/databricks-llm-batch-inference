import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Tuple, Optional, List

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

from llm_batch_inference.inference.engine import InferenceEngine
from llm_batch_inference.config.inference_config import InferenceConfig
from llm_batch_inference.api.openai_client import OpenAIClient
from llm_batch_inference.utils.logger import Logger
from llm_batch_inference.inference.batch_processor import BatchProcessor, BatchInference

@pytest.fixture
def mock_config():
    config = MagicMock(spec=InferenceConfig)
    config.concurrency = 2
    config.enable_logging = False
    config.llm_task = "completion"  # Add this line
    config.logging_interval = 40
    config.request_params = {"temperature": 0, "max_tokens": 500}
    config.endpoint = "databricks-meta-llama-3-1-70b-instruct"
    config.timeout = 300
    config.max_retries_backpressure = 3
    config.max_retries_other = 3
    config.prompt = ""
    return config

from databricks.connect import DatabricksSession

@pytest.fixture(scope="module")
def spark():
    return DatabricksSession.builder.getOrCreate()

from unittest.mock import patch, MagicMock
from databricks.connect import DatabricksSession

@pytest.fixture(scope="module")
def mock_databricks_session():
    with patch('databricks.connect.DatabricksSession') as mock_session:
        mock_instance = MagicMock()
        mock_session.builder.getOrCreate.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_engine():
    return MagicMock(spec=InferenceEngine)

@pytest.fixture
def processor(mock_engine, mock_config):
    return BatchProcessor(mock_engine, mock_config)

@pytest.fixture
def batch_inference(mock_config):
    with patch('llm_batch_inference.api.openai_client.OpenAIClient') as mock_client:
        with patch('llm_batch_inference.inference.engine.InferenceEngine') as mock_engine:
            with patch('llm_batch_inference.utils.logger.Logger') as mock_logger:
                return BatchInference(mock_config, "fake_token", "fake_root")

@pytest.fixture(scope="module")
def spark():
    return SparkSession.builder.appName("TestBatchProcessor").getOrCreate()

def test_process_item(processor):
    processor.engine.infer.return_value = ("processed text", 5)
    result = processor.process_item((1, "test text"))
    assert result == (1, "processed text", 5, None)

def test_process_item_exception(processor):
    processor.engine.infer.side_effect = Exception("Test error")
    result = processor.process_item((1, "test text"))
    assert result == (1, None, 0, "Test error")

@pytest.mark.asyncio
async def test_process_item_async(processor):
    processor.engine.async_infer = AsyncMock(return_value=("processed text", 5))
    result = await processor.process_item_async((1, "test text"))
    assert result == (1, "processed text", 5, None)

@pytest.mark.asyncio
async def test_process_item_async_exception(processor):
    processor.engine.async_infer = AsyncMock(side_effect=Exception("Test error"))
    result = await processor.process_item_async((1, "test text"))
    assert result == (1, None, 0, "Test error")

def test_process_batch(processor):
    processor.engine.infer.side_effect = [("processed1", 5), ("processed2", 6)]
    results = processor.process_batch([(1, "text1"), (2, "text2")])
    assert results == [(1, "processed1", 5, None), (2, "processed2", 6, None)]

@pytest.mark.asyncio
async def test_process_batch_async(processor):
    processor.engine.async_infer = AsyncMock(side_effect=[("processed1", 5), ("processed2", 6)])
    results = await processor.process_batch_async([(1, "text1"), (2, "text2")])
    assert results == [(1, "processed1", 5, None), (2, "processed2", 6, None)]

def test_run_batch_inference(batch_inference):
    # Mock the run_batch_inference_async method
    batch_inference.run_batch_inference_async = AsyncMock(return_value=[(1, "processed", 5, None)])
    
    # Mock asyncio.run to return the result of run_batch_inference_async directly
    with patch('asyncio.run', new=lambda x: x):
        results = batch_inference.run_batch_inference([(1, "test")])
    
    assert results == [(1, "processed", 5, None)]
    assert batch_inference.nest_asyncio_applied == True
    batch_inference.run_batch_inference_async.assert_called_once_with([(1, "test")])

@pytest.mark.asyncio
async def test_run_batch_inference_async(batch_inference):
    batch_inference.processor.process_batch_async = AsyncMock(return_value=[(1, "processed", 5, None)])
    results = await batch_inference.run_batch_inference_async([(1, "test")])
    assert results == [(1, "processed", 5, None)]

@pytest.fixture
def batch_inference():
    with patch('llm_batch_inference.inference.batch_processor.OpenAIClient'):
        with patch('llm_batch_inference.inference.batch_processor.InferenceEngine'):
            from llm_batch_inference.inference.batch_processor import BatchInference, InferenceConfig
            config = InferenceConfig()
            return BatchInference(config, "fake_token", "fake_root")

# def test_run_batch_inference_pandas_udf(batch_inference):
#     # Mock DataFrame
#     mock_df = MagicMock()
#     mock_df.__getitem__.return_value = MagicMock()
#     mock_df.withColumn.return_value = mock_df

#     # Create schema
#     schema = StructType([
#         StructField("output", StringType(), True),
#         StructField("tokens", IntegerType(), True),
#         StructField("error", StringType(), True)
#     ])

#     # Mock pandas_udf
#     with patch('llm_batch_inference.inference.batch_processor.pandas_udf') as mock_pandas_udf:
#         # Create a mock UDF that returns another mock when called
#         mock_udf = MagicMock()
#         mock_udf_result = MagicMock()
#         mock_udf.return_value = mock_udf_result
#         mock_pandas_udf.return_value = mock_udf

#         # Call the method
#         result_df = batch_inference.run_batch_inference_pandas_udf(
#             mock_df, "input", ["output", "tokens", "error"], schema
#         )

#         # Assertions
#         mock_pandas_udf.assert_called_once_with(schema)
#         mock_udf.assert_called_once_with(mock_df["input"])
#         mock_df.withColumn.assert_called_once_with("result", mock_udf_result)

#         assert result_df == mock_df

# def test_run_batch_inference_pandas_udf(batch_inference, mock_databricks_session):
#     input_data = [("text1",), ("text2",)]
#     input_df = mock_databricks_session.createDataFrame(input_data, ["input"])
    
#     output_schema = StructType([
#         StructField("output", StringType(), True),
#         StructField("tokens", IntegerType(), True),
#         StructField("error", StringType(), True)
#     ])
    
#     mock_result_df = mock_databricks_session.createDataFrame([("processed1", 5, None), ("processed2", 6, None)], 
#                                            ["output", "tokens", "error"])
    
#     batch_inference.processor.process_with_pandas_udf = MagicMock(return_value=mock_result_df)
    
#     result_df = batch_inference.run_batch_inference_pandas_udf(input_df, "input", ["output", "tokens", "error"], output_schema)
    
#     # Since we're using a mock, we can't use DataFrame methods directly
#     # Instead, we'll check if the mock methods were called correctly
#     batch_inference.processor.process_with_pandas_udf.assert_called_once()
#     assert result_df == mock_result_df

def test_ensure_nest_asyncio(batch_inference):
    batch_inference._ensure_nest_asyncio()
    assert batch_inference.nest_asyncio_applied == True
    
    # Calling it again should not raise an error
    batch_inference._ensure_nest_asyncio()

if __name__ == "__main__":
    pytest.main()