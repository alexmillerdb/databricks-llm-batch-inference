import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pytest
import asyncio
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from llm_batch_inference.config.inference_config import InferenceConfig
from llm_batch_inference.inference.batch_processor import BatchInference

# Setup SparkSession for pandas UDF tests
@pytest.fixture(scope="module")
def spark():
    return SparkSession.builder.appName("IntegrationTests").getOrCreate()

# Setup BatchInference instance
@pytest.fixture(scope="module")
def batch_inference():
    config = InferenceConfig(
        endpoint="test-endpoint",
        concurrency=2,
        llm_task="completion"
    )
    return BatchInference(config, API_TOKEN="fake_token", API_ROOT="fake_root")

# Async integration test
@pytest.mark.asyncio
async def test_run_batch_inference_async(batch_inference):
    texts_with_index = [
        (1, "Hello, world!"),
        (2, "Testing async batch inference"),
        (3, "Integration test")
    ]
    
    results = await batch_inference.run_batch_inference_async(texts_with_index)
    
    assert len(results) == 3
    for result in results:
        assert len(result) == 4
        assert isinstance(result[0], int)
        assert isinstance(result[1], str) or result[1] is None
        assert isinstance(result[2], int)
        assert isinstance(result[3], str) or result[3] is None

# Synchronous integration test
def test_run_batch_inference(batch_inference):
    texts_with_index = [
        (1, "Hello, world!"),
        (2, "Testing sync batch inference"),
        (3, "Integration test")
    ]
    
    results = batch_inference.run_batch_inference(texts_with_index)
    
    assert len(results) == 3
    for result in results:
        assert len(result) == 4
        assert isinstance(result[0], int)
        assert isinstance(result[1], str) or result[1] is None
        assert isinstance(result[2], int)
        assert isinstance(result[3], str) or result[3] is None

# Pandas UDF integration test
def test_run_batch_inference_pandas_udf(batch_inference, spark):
    # Create a sample DataFrame
    data = [("Hello, world!",), ("Testing pandas UDF",), ("Integration test",)]
    df = spark.createDataFrame(data, ["input_text"])
    
    # Define output schema
    output_schema = StructType([
        StructField("output_text", StringType(), True),
        StructField("token_count", IntegerType(), True),
        StructField("error", StringType(), True)
    ])
    
    # Run batch inference with pandas UDF
    result_df = batch_inference.run_batch_inference_pandas_udf(
        df, 
        input_col="input_text", 
        output_cols=["output_text", "token_count", "error"],
        schema=output_schema
    )
    
    # Check results
    assert result_df.count() == 3
    assert "output_text" in result_df.columns
    assert "token_count" in result_df.columns
    assert "error" in result_df.columns
    
    # Collect results and check types
    results = result_df.collect()
    for row in results:
        assert isinstance(row["output_text"], str) or row["output_text"] is None
        assert isinstance(row["token_count"], int)
        assert isinstance(row["error"], str) or row["error"] is None

# Test error handling
@pytest.mark.asyncio
async def test_run_batch_inference_async_with_error(batch_inference):
    texts_with_index = [
        (1, "Valid input"),
        (2, ""),  # Empty input to potentially trigger an error
        (3, "Another valid input")
    ]
    
    results = await batch_inference.run_batch_inference_async(texts_with_index)
    
    assert len(results) == 3
    for result in results:
        if result[0] == 2:  # The empty input
            assert result[1] is None
            assert result[2] == 0
            assert result[3] is not None  # Should contain an error message
        else:
            assert result[1] is not None
            assert result[2] > 0
            assert result[3] is None

# Test concurrency
@pytest.mark.asyncio
async def test_concurrency(batch_inference):
    # Create a large batch of inputs
    texts_with_index = [(i, f"Test input {i}") for i in range(100)]
    
    start_time = time.time()
    results = await batch_inference.run_batch_inference_async(texts_with_index)
    end_time = time.time()
    
    assert len(results) == 100
    
    # Check if the processing time is reasonable given the concurrency setting
    # This is a rough check and may need adjustment based on the actual processing time
    expected_time = (100 / batch_inference.config.concurrency) * 0.1  # Assuming each inference takes about 0.1 seconds
    assert (end_time - start_time) < expected_time * 1.5  # Allow some buffer
