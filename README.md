# Databricks LLM batch inference with FM APIs, Provisioned Throughput, & External Models

## Overview
The goal of this project is to simplify batch inference with LLMs through simple API interface for Databricks GenAI model serving. This follows a simple process:

1. Create a Delta table that includes the text you want to run inference on.
2. Define the data configurations such as table path
3. Define the inference congirations such as endpoint, prompt, concurrency, and LLM task (completion or task)
4. Run the batch inference based on configurations

## Features
- Simplified API that allows you to iterate quickly without having to worry about writing Python async or Spark UDF commands
- Built-in retry logic on the client-side for specific error status codes such as 429 and 503
- Simple developer experience that allows users to test and measure different prompts and concurrent requests to meet SLAs (throughput/latency) and model performance
- Examples for specific LLM use cases such as fine-tuning LLM using Mosaic fine-tuning for structured outputs

## Installation

To install, please run at top of Databricks notebook. Refer to https://github.com/alexmillerdb/databricks-llm-batch-inference/blob/main/examples/batch_inference/batch-inference-example.py 

```python

%pip install git+https://github.com/alexmillerdb/databricks-llm-batch-inference.git

```

## Notice

## Supported Frameworks

## Running LLM batch inference using Python 


```python
import mlflow

from llm_batch_inference.config.data_processor_config import DataProcessorConfig
from llm_batch_inference.config.inference_config import InferenceConfig
from llm_batch_inference.inference.batch_processor import BatchProcessor, BatchInference
from llm_batch_inference.data.processor import DataProcessor

data_config = DataProcessorConfig(
    input_table_name="alex_m.gen_ai.news_qa_summarization",
    input_column_name="prompt",
    input_num_rows=100  # Optional
)

# Inference configuration
inference_config = InferenceConfig(
    endpoint="databricks-meta-llama-3-1-70b-instruct",  # any FM API/PT endpoint
    timeout=300,
    max_retries_backpressure=3,
    max_retries_other=3,
    prompt="", # if you pass prompt it will dynamically create prompt within API client
    request_params={"temperature": 0, "max_tokens": 500},
    concurrency=5,
    logging_interval=40,
    llm_task="chat", # task
    enable_logging=False
)

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
results = batch_inference.run_batch_inference(texts_with_index)

print(results)
assert len(results) == data_config.input_num_rows, "Results length does not match the data input"
print("Batch inference completed successfully")

```

## Running LLM batch inference using Spark

```python
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StructType, StructField, StringType, BooleanType

import mlflow

from llm_batch_inference.config.data_processor_config import DataProcessorConfig
from llm_batch_inference.config.inference_config import InferenceConfig
from llm_batch_inference.inference.batch_processor import BatchProcessor, BatchInference
from llm_batch_inference.data.processor import DataProcessor

# Data processing configuration
data_config = DataProcessorConfig(
    input_table_name="alex_m.gen_ai.news_qa_summarization",
    input_column_name="prompt",
    input_num_rows=100  # Optional
)

# Inference configuration
inference_config = InferenceConfig(
    endpoint="databricks-meta-llama-3-1-70b-instruct",  # any FM API/PT endpoint
    timeout=300,
    max_retries_backpressure=3,
    max_retries_other=3,
    prompt="", # if you pass prompt it will dynamically create prompt within API client
    request_params={"temperature": 0, "max_tokens": 500},
    concurrency=5,
    logging_interval=40,
    llm_task="chat", # task
    enable_logging=False
)

# Get API_ROOT and API_TOKEN
API_ROOT = mlflow.utils.databricks_utils.get_databricks_host_creds().host
API_TOKEN = mlflow.utils.databricks_utils.get_databricks_host_creds().token

# Read input data
spark_df = spark.table(data_config.input_table_name) \
    .limit(data_config.input_num_rows) \
    .repartition(inference_config.concurrency)

# Create BatchInference
batch_inference = BatchInference(inference_config, API_TOKEN, API_ROOT)

# Run batch inference
print("Running batch inference")
schema = StructType([
    StructField("output_text", StringType(), True),
    StructField("token_count", IntegerType(), True),
    StructField("error", StringType(), True)
])
results = batch_inference.run_batch_inference_pandas_udf(df=spark_df, 
                                                         input_col=data_config.input_column_name, 
                                                         output_cols=["output_text", "token_count", "error"], 
                                                         schema=schema).cache()

display(results)
assert results.count() == data_config.input_num_rows, "Results length does not match the data input"
print("Batch inference completed successfully")

```
