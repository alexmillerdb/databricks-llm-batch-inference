# Databricks notebook source
# MAGIC %pip install -r ../../requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## Install whl file on cluster

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.compute import Library

w = WorkspaceClient()

# Specify the cluster id where you want to install the library
cluster_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterId")

# Specify the path to your .whl file
whl_path = "/Volumes/alex_m/gen_ai/llm_batch_inference_whl/llm_batch_inference-0.1.1-py3-none-any.whl"

# Install the .whl file as a library on the cluster
# w.libraries.install()
w.libraries.install(cluster_id, [Library(whl=whl_path)])

# COMMAND ----------

# MAGIC %md ## Import libraries and custom python package (from whl file)

# COMMAND ----------

import os
import json
import re
import asyncio
from pyspark.sql import functions as F
from pyspark.sql.functions import lit
from pyspark.sql.types import IntegerType, StructType, StructField, StringType, BooleanType

from mlflow.utils.databricks_utils import get_databricks_host_creds
import mlflow

from databricks.sdk import WorkspaceClient

from llm_batch_inference.config.data_processor_config import DataProcessorConfig
from llm_batch_inference.config.inference_config import InferenceConfig
from llm_batch_inference.inference.batch_processor import BatchProcessor, BatchInference
from llm_batch_inference.data.processor import DataProcessor

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch inference client & API
# MAGIC
# MAGIC The following code:
# MAGIC - Create `data_config` and `inference_config`
# MAGIC - Setup `DataProcessor` to create tuples of (index, prompt)
# MAGIC - Setup the `BatchInference` asynchronous client 
# MAGIC - Run batch inference using asyncio

# COMMAND ----------

# DBTITLE 1,Setup data and inference config
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

# COMMAND ----------

# DBTITLE 1,Process text data and run batch inference
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
# MAGIC ## Extract JSON responses, join responses to source dataframe and write to UC

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

# DBTITLE 1,Calculate Incorrect JSON outputs
null_ct = output_sdf.filter(F.col("resp_chat_parsed").isNull()).count()
print(f"Total null ct: {null_ct}")
print(f"Percent null ct: {null_ct / output_sdf.count()}")

# COMMAND ----------

ft_dataset = output_sdf.filter(F.col("resp_chat_parsed").isNotNull()) \
  .select(F.col("prompt"), F.to_json(F.col("resp_chat_parsed")).alias("response"))

ft_dataset.display()
assert ft_dataset.filter(F.col("response").isNull()).count() == 0, "Null response found in dataset"
ft_dataset.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("alex_m.gen_ai.news_qa_summarization_llm_ft_dataset")
