# Databricks notebook source
# MAGIC %pip install -r requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import json
import httpx
import re
import asyncio
import time
from tenacity import (
    retry,
    stop_after_attempt,
    retry_if_exception,
    wait_random_exponential,
)
from pyspark.sql import functions as F
from pyspark.sql.functions import lit
from pyspark.sql.types import IntegerType, StructType, StructField, StringType, BooleanType

from mlflow.utils.databricks_utils import get_databricks_host_creds
import mlflow

import openai
from openai import OpenAI
from databricks.sdk import WorkspaceClient

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch inference client & API
# MAGIC
# MAGIC The following code:
# MAGIC - Sets up the asynchronous client 
# MAGIC - Defines credentials for accessing the batch inference endpoint
# MAGIC - Defines the prompt that is used for batch inference

# COMMAND ----------

import sys
import os
from databricks.connect import DatabricksSession
# from pyspark.dbutils import DBUtils

# # Get the current notebook path
spark = DatabricksSession.builder.getOrCreate()
current_directory = os.getcwd()
root_directory = os.path.normpath(os.path.join(current_directory, '..'))
sys.path.append(current_directory)
sys.path.append(root_directory)

# COMMAND ----------

import asyncio
import mlflow
from pyspark.sql import SparkSession
import nest_asyncio

nest_asyncio.apply()

from src.config.data_processor_config import DataProcessorConfig
from src.config.inference_config import InferenceConfig
from src.inference.batch_processor import BatchProcessor, BatchInference
from src.data.processor import DataProcessor

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run batch inference

# COMMAND ----------

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
    llm_task="chat", # task
    enable_logging=False
)

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Join responses to source dataframe and write to UC

# COMMAND ----------

# Define the Python function to extract and parse JSON from text
def extract_json(text):
    try:
        # Regular expression to extract the JSON part between ```json\n and \n```
        json_match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
        if json_match:
            json_string = json_match.group(1)
            # Parse the extracted JSON string
            json_data = json.loads(json_string)
            # Return the JSON string to be stored in the DataFrame
            return json.dumps(json_data)  # Returning as string
        else:
            return None
    except Exception as e:
        return None
      
def is_json_valid(json_str):
    try:
        json.loads(json_str)
        return True
    except ValueError:
        return False

# Register the Python function as a Spark UDF
extract_json_udf = F.udf(extract_json, StringType())
is_json_valid_udf = F.udf(is_json_valid, BooleanType())

# COMMAND ----------

# Step 1: Create a DataFrame from model responses with indices
schema = f"{index_column} long, resp_chat string, resp_total_tokens int, resp_error string"
# Define schema for structured response
json_schema = "struct<summary:string, sentiment:string, topic:string>"

# Dynamically check if the UDF needs to be applied or not
response_sdf = spark.createDataFrame(results, schema=schema) \
    .withColumn(
        "resp_chat_clean",
        F.when(is_json_valid_udf("resp_chat"), F.col("resp_chat"))  # If already structured, use as is
         .otherwise(extract_json_udf(F.col("resp_chat")))        # Otherwise, apply the UDF
    ) \
    .withColumn("resp_chat_parsed", F.from_json(F.col("resp_chat_clean"), schema=json_schema)) \
    .drop("resp_chat_clean")

# Step 2: Join the DataFrames on the index column
output_sdf = source_sdf.join(response_sdf, on=index_column).drop(index_column)
output_sdf.display()
# Step 3: Persistent the final spark dataframe into UC output table
output_sdf.write.mode("overwrite").option("mergeSchema", "true").saveAsTable("alex_m.gen_ai.news_qa_summarization_llm_output")

# COMMAND ----------

null_ct = output_sdf.filter(F.col("resp_chat_parsed").isNull()).count()
print(f"Total null ct: {null_ct}")
print(f"Percent null ct: {null_ct / output_sdf.count()}")
