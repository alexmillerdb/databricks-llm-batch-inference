# Databricks notebook source
# Install openai package
%pip install httpx==0.27.0
%pip install mlflow==2.11.1
%pip install tenacity==8.2.3
dbutils.library.restartPython()

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

# DBTITLE 1,Batch Inference client API
# 1. Config Management
from pydantic import BaseModel, Field, validator
from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
import asyncio
import time

# 
class InferenceConfig(BaseModel):
    endpoint: str = Field(default="databricks-meta-llama-3-1-70b-instruct")
    timeout: int = Field(default=300)
    max_retries_backpressure: int = Field(default=3)
    max_retries_other: int = Field(default=3)
    prompt: Optional[str] = Field(default=None)
    request_params: Dict = Field(default_factory=dict)
    concurrency: int = Field(default=15)
    logging_interval: int = Field(default=40)
    enable_logging: bool = Field(default=True)
    llm_task: str = Field(default="", choices=["chat", "completion"])

    @validator('endpoint', 'llm_task', always=True)
    def check_required_fields(cls, v, field):
        assert v, f"{field.name} is required and cannot be empty"
        return v

    @validator('request_params', always=True)
    def check_request_params(cls, v):
        assert v, "request_params is required and cannot be empty"
        return v

# 2. API Client
class APIClientInterface(ABC):
    @abstractmethod
    async def predict(self, text: str) -> Tuple[str, int]:
        pass

class OpenAIClient(APIClientInterface):
    def __init__(self, config: InferenceConfig):
        self.config = config
        # Initialize OpenAI client
        API_ROOT = mlflow.utils.databricks_utils.get_databricks_host_creds().host
        API_TOKEN = mlflow.utils.databricks_utils.get_databricks_host_creds().token

        self.client = OpenAI(
            api_key=API_TOKEN,
            base_url=f"{API_ROOT}/serving-endpoints"
        )

    async def predict(self, text: str) -> Tuple[str, int]:
        # If the model is chat-based, use the ChatCompletion API
        if self.config.llm_task == "chat":
            messages = [{"role": "user", "content": self.config.prompt + str(text) if self.config.prompt else str(text)}]
            try:
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.config.endpoint,
                    messages=messages,
                    **self.config.request_params
                )
                content = response.choices[0].message.content
                total_tokens = response.usage.total_tokens
                return content, total_tokens
            except Exception as e:
                print(f"Error while making OpenAI ChatCompletion API call: {e}")
                raise

        # If the model expects plain completion (non-chat)
        elif self.config.llm_task == "completion":
            try:
                response = await asyncio.to_thread(
                    self.client.completions.create,
                    model=self.config.endpoint,
                    prompt=self.config.prompt + str(text) if self.config.prompt else str(text),
                    **self.config.request_params
                )
                content = response.choices[0].text
                total_tokens = response.usage.total_tokens
                return content, total_tokens
            except Exception as e:
                print(f"Error while making OpenAI Completion API call: {e}")
                raise

# 3. Inference Engine
class InferenceEngine:
    def __init__(self, client: APIClientInterface):
        self.client = client

    async def infer(self, text: str) -> Tuple[str, int]:
        return await self.client.predict(text)
      
# 4. Batch Processing
class BatchProcessor:
    def __init__(self, engine: InferenceEngine, config: InferenceConfig):
        self.engine = engine
        self.config = config

    async def process_item(self, item: Tuple[int, str]) -> Tuple[int, Optional[str], int, Optional[str]]:
        index, text = item
        try:
            content, num_tokens = await self.engine.infer(text)
            return (index, content, num_tokens, None)
        except Exception as e:
            return (index, None, 0, str(e))

    async def process_batch(self, items: List[Tuple[int, str]]) -> List[Tuple[int, Optional[str], int, Optional[str]]]:
        semaphore = asyncio.Semaphore(self.config.concurrency)
        async def process_with_semaphore(item):
            async with semaphore:
                return await self.process_item(item)
        return await asyncio.gather(*[process_with_semaphore(item) for item in items])
      
# 5. Error Handling
def is_backpressure(error: httpx.HTTPStatusError):
    if hasattr(error, "response") and hasattr(error.response, "status_code"):
        return error.response.status_code in (429, 503)

def is_other_error(error: httpx.HTTPStatusError):
    if hasattr(error, "response") and hasattr(error.response, "status_code"):
        return error.response.status_code != 503 and (
            error.response.status_code >= 500 or error.response.status_code == 408)
        
# 6. Logging
class Logger:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.counter = 0
        self.start_time = time.time()
        self.enable_logging = config.enable_logging

    def log_progress(self):
        if not self.enable_logging:
            return
        self.counter += 1
        if self.counter % self.config.logging_interval == 0:
            elapsed = time.time() - self.start_time
            print(f"Processed {self.counter} requests in {elapsed:.2f} seconds.")
    
    def log_total_time(self, total_items: int):
        if not self.enable_logging:
            return
        total_time = time.time() - self.start_time
        print(f"Total processing time: {total_time:.2f} seconds for {total_items} items.")
        print(f"Average time per item: {total_time/total_items:.4f} seconds.")

# Main class tying it all together
class BatchInference:
    def __init__(self, config: InferenceConfig):
        self.config = config
        client = OpenAIClient(config)
        self.engine = InferenceEngine(client)
        self.processor = BatchProcessor(self.engine, config)
        self.logger = Logger(config)

    async def __call__(self, texts_with_index: List[Tuple[int, str]]) -> List[Tuple[int, Optional[str], int, Optional[str]]]:
        self.logger.start_time = time.time()  # Reset start time
        results = await self.processor.process_batch(texts_with_index)
        for _ in results:
            self.logger.log_progress()
        self.logger.log_total_time(len(texts_with_index))
        return results

# COMMAND ----------

import uuid
from typing import List, Tuple, Optional
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import monotonically_increasing_id
import pandas as pd

class DataProcessorConfig:
    def __init__(self, input_table_name: str, input_column_name: str, input_num_rows: Optional[int] = None):
        self.input_table_name = input_table_name
        self.input_column_name = input_column_name
        self.input_num_rows = input_num_rows

class DataProcessor:
    def __init__(self, spark: SparkSession, config: DataProcessorConfig):
        self.spark = spark
        self.config = config
        self.index_column = f"index_{uuid.uuid4().hex[:4]}"
        self.source_sdf: Optional[DataFrame] = None
        self.input_sdf: Optional[DataFrame] = None
        self.texts_with_index: Optional[List[Tuple[int, str]]] = None

    def load_spark_dataframe(self) -> DataFrame:
        """Load the Spark DataFrame from the input table and add an index column."""
        self.source_sdf = (self.spark.table(self.config.input_table_name)
                           .withColumn(self.index_column, monotonically_increasing_id()))
        return self.source_sdf

    def select_and_limit_data(self, sdf: DataFrame) -> DataFrame:
        """Select required columns and optionally limit the number of rows."""
        result = sdf.select(self.index_column, self.config.input_column_name)
        if self.config.input_num_rows:
            result = result.limit(self.config.input_num_rows)
        self.input_sdf = result
        return self.input_sdf

    def convert_to_list(self, sdf: DataFrame) -> List[Tuple[int, str]]:
        """Convert Spark DataFrame to a list of tuples."""
        pandas_df = sdf.toPandas()
        return [row for row in pandas_df.itertuples(index=False, name=None)]

    def process(self) -> List[Tuple[int, str]]:
        """Main method to process the data."""
        self.source_sdf = self.load_spark_dataframe()
        self.input_sdf = self.select_and_limit_data(self.source_sdf)
        self.texts_with_index = self.convert_to_list(self.input_sdf)
        return self.texts_with_index

    def get_source_sdf(self) -> Optional[DataFrame]:
        """Get the source Spark DataFrame."""
        return self.source_sdf

    def get_input_sdf(self) -> Optional[DataFrame]:
        """Get the input Spark DataFrame."""
        return self.input_sdf

    def get_texts_with_index(self) -> Optional[List[Tuple[int, str]]]:
        """Get the processed list of texts with index."""
        return self.texts_with_index

# Usage
spark = SparkSession.builder.getOrCreate()  # You would typically get this from your Spark environment
data_config = DataProcessorConfig(
    input_table_name="alex_m.gen_ai.news_qa_summarization",
    input_column_name="prompt",
    input_num_rows=1000  # Optional
)

processor = DataProcessor(spark, data_config)
texts_with_index = processor.process()

# Now you can access the DataFrames and the processed list
source_sdf = processor.get_source_sdf()
input_sdf = processor.get_input_sdf()
texts_with_index = processor.get_texts_with_index()
index_column = processor.index_column

# You can use these variables as needed
if source_sdf:
    print("Source DataFrame count:", source_sdf.count())
if input_sdf:
    print("Input DataFrame count:", input_sdf.count())
if texts_with_index:
    print("Number of processed texts:", len(texts_with_index))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Run batch inference

# COMMAND ----------

# DBTITLE 1,Run Batch Inference on "chat" endpoint
inference_config = InferenceConfig(
    endpoint="databricks-meta-llama-3-1-70b-instruct",
    timeout=300,
    max_retries_backpressure=3,
    max_retries_other=3,
    prompt="",
    request_params={"temperature": 0.7, "max_tokens": 100},
    concurrency=15,
    logging_interval=40,
    llm_task="chat",
    enable_logging=False
)

batch_inference = BatchInference(inference_config)
results = await batch_inference(texts_with_index)
results

# COMMAND ----------

# DBTITLE 1,Run inference on "completion" endpoint
inference_config = InferenceConfig(
    endpoint="llama_8b_instruct_structured_outputs",
    timeout=300,
    max_retries_backpressure=3,
    max_retries_other=3,
    prompt="",
    request_params={"temperature": 0.7, "max_tokens": 100},
    concurrency=15,
    logging_interval=40,
    llm_task="completion",
    enable_logging=True # TO DO fix logging code
)

batch_inference = BatchInference(inference_config)
results = await batch_inference(texts_with_index)
results

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
