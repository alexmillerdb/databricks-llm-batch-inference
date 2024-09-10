# MAGIC %pip install -r requirements.txt
# MAGIC dbutils.library.restartPython()

import sys
import os
from databricks.connect import DatabricksSession
# from pyspark.dbutils import DBUtils

# # Get the current notebook path
spark = DatabricksSession.builder.getOrCreate()
current_directory = os.getcwd()
root_directory = os.path.normpath(os.path.join(current_directory, '..'))
sys.path.append(root_directory)

# # Verify the paths
# print(sys.path)
# print(f"Current directory: {current_directory}")

import asyncio
import mlflow
from pyspark.sql import SparkSession
import nest_asyncio

nest_asyncio.apply()

from config.data_processor_config import DataProcessorConfig
from config.inference_config import InferenceConfig
from inference.batch_processor import BatchProcessor, BatchInference
from data.processor import DataProcessor

# Data processing configuration
data_config = DataProcessorConfig(
    input_table_name="alex_m.gen_ai.news_qa_summarization",
    input_column_name="prompt",
    input_num_rows=100  # Optional
)

# Inference configuration
inference_config = InferenceConfig(
    endpoint="databricks-meta-llama-3-1-70b-instruct",
    timeout=300,
    max_retries_backpressure=3,
    max_retries_other=3,
    prompt="",
    request_params={"temperature": 0.1, "max_tokens": 100},
    concurrency=5,
    logging_interval=40,
    llm_task="chat",
    enable_logging=False
)

if __name__ == "__main__":
    # spark = SparkSession.builder.getOrCreate()
    
    # Get API_ROOT and API_TOKEN
    API_ROOT = mlflow.utils.databricks_utils.get_databricks_host_creds().host
    API_TOKEN = mlflow.utils.databricks_utils.get_databricks_host_creds().token
    
    processor = DataProcessor(spark, data_config)
    texts_with_index = processor.process()

    # Now you can access the DataFrames and the processed list
    source_sdf = processor.get_source_sdf()
    input_sdf = processor.get_input_sdf()
    texts_with_index = processor.get_texts_with_index()
    index_column = processor.index_column

    # Print information about the processed data
    assert source_sdf, "Source DataFrame is not available"
    if source_sdf:
        print("Source DataFrame count:", source_sdf.count())
    if input_sdf:
        print("Input DataFrame count:", input_sdf.count())
    if texts_with_index:
        print("Number of processed texts:", len(texts_with_index))

    # Create BatchInference
    batch_inference = BatchInference(inference_config, API_TOKEN, API_ROOT)

    # Run batch inference
    print("Running batch inference")
    results = asyncio.run(batch_inference.run_batch_inference(texts_with_index))

    print(results)
    assert len(results) == data_config.input_num_rows, "Results length does not match the data input"
    print("Batch inference completed successfully")