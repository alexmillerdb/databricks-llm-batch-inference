# Databricks notebook source
# MAGIC %md
# MAGIC # Chat model batch inference tasks using Python
# MAGIC
# MAGIC This notebook is the partner notebook to the Perform batch inference on a provisioned throughput endpoint notebook. This notebook and the **Perform batch inference on a provisioned throughput endpoint** notebook must be in the same directory of your workspace for the batch inference workflow to perform successfully.
# MAGIC
# MAGIC The following tasks are accomplished in this notebook: 
# MAGIC
# MAGIC 1. Read data from the input table and input column
# MAGIC 2. Construct the requests and send the requests to a Foundation Model APIs endpoint with some kind of concurrency
# MAGIC 3. Persist input row together with the response data to the output table

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install packages

# COMMAND ----------

# DBTITLE 1,Install Packages
# Install openai package
%pip install httpx==0.27.0
%pip install mlflow==2.11.1
%pip install tenacity==8.2.3
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up configuration parameters 
# MAGIC

# COMMAND ----------

# catalog = "main"
# schema = "alex_m"
# endpoint = "meta_llama_3_8b_instruct_frontier"
# alex_m.gen_ai.news_qa_summarization

# COMMAND ----------

# DBTITLE 1,Configurations
# Client Configuration
dbutils.widgets.text("endpoint", "databricks-dbrx-instruct", "Endpoint")
dbutils.widgets.text("timeout", "300", "Timeout")
dbutils.widgets.text("max_retries_backpressure", "20", "#Max Retries (backpressure)")
dbutils.widgets.text("max_retries_other", "5", "#Max Retries (other error)")

# Client Request Configuration
dbutils.widgets.text("prompt", "Can you tell me the name of the US state that serves the provided ZIP code? zip code: ", "Prompt (system)")
dbutils.widgets.text("request_params", '{"max_tokens": 1000, "temperature": 0}', "Chat Request Params")

# Batch Inference Configuration
dbutils.widgets.text("concurrency", "15", "#Concurrency Requests")

# Table Configuration
dbutils.widgets.text("input_table_name", "samples.nyctaxi.trips", "Input Table Name")
dbutils.widgets.text("input_column_name", "pickup_zip", "Input Column Name")
dbutils.widgets.text("input_num_rows", "100", "Input Number Rows")
dbutils.widgets.text("output_table_name", "main.default.nyctaxi_trips_llm_output", "Output Table Name")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parameters
# MAGIC
# MAGIC   | Parameter Name      | Example Value | Description | 
# MAGIC   | --------------- | ---------------------- | ------------------------------- | 
# MAGIC   | `endpoint` | databricks-dbrx-instruct | The name of the Databricks Serving endpoint. You can find the endpoint name under the `Serving` tab. | 
# MAGIC   | `prompt` | Can you tell me the name of the US state that serves the provided ZIP code? | The system prompt to use in the batch inference. | 
# MAGIC   | `input_table_name` | samples.nyctaxi.trips | The name of the input table in Unity Catalog. | 
# MAGIC   | `input_column_name` | pickup_zip | The name of the column in the input table to use.   | 
# MAGIC   | `input_num_rows` | 1000 | The number of rows to process for the input table. |
# MAGIC   | `output_table_name` | main.default.nyctaxi_trips_llm_output | The name of the output table in Unity Catalog. All of the input columns and the inference result columns (such as `chat`, `total_tokens`, `error_message`) are persisted in the output table.  |
# MAGIC
# MAGIC ## Addtional parameters
# MAGIC   | Parameter Name      | Example Value | Description | 
# MAGIC   | --------------- | ---------------------- | ------------------------------- | 
# MAGIC   | `concurrency` | 15 | The number of concurrent requests send to server. | 
# MAGIC   | `timeout` | 300 | The timeout for an HTTP request on the client side |
# MAGIC   | `max_retries_backpressure` | 20 | The maximum number of retries due to a backpressure status code (such as 429 or 503). | 
# MAGIC   | `max_retries_other` | 5 | The maximum number of retries due to non-backpressure status code (such as 5xx, 408, or 409). | 
# MAGIC   | `request_params` | {"max_tokens": 1000, "temperature": 0} | The extra chat http request parameters in json format (reference: https://docs.databricks.com/en/machine-learning/foundation-models/api-reference.html#chat-request) | 

# COMMAND ----------

# DBTITLE 1,Load Configurations
import json

# Load configurations from widgets
config_endpoint = dbutils.widgets.get("endpoint")
config_timeout = int(dbutils.widgets.get("timeout"))
config_max_retries_backpressure = int(dbutils.widgets.get("max_retries_backpressure"))
config_max_retries_other = int(dbutils.widgets.get("max_retries_other"))

config_prompt = dbutils.widgets.get("prompt")
config_request_params = json.loads(
    dbutils.widgets.get("request_params")
 ) # Reference: https://docs.databricks.com/en/machine-learning/foundation-models/api-reference.html#chat-request

config_concurrecy = int(dbutils.widgets.get("concurrency"))
config_logging_interval = 20

config_input_table = dbutils.widgets.get("input_table_name")
config_input_column = dbutils.widgets.get("input_column_name")
config_input_num_rows = dbutils.widgets.get("input_num_rows")
if config_input_num_rows:
    config_input_num_rows = int(config_input_num_rows)
config_output_table = dbutils.widgets.get("output_table_name")

# COMMAND ----------

print(f"endpoint: {config_endpoint}")
print(f"config prompt: {config_prompt}")
print(f"config request parameters: {config_request_params}")
print(f"concurrency: {config_concurrecy}")
print(f"config input table: {config_input_table}")
print(f"config input column: {config_input_column}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch inference client & API
# MAGIC
# MAGIC The following code:
# MAGIC - Sets up the asynchronous client 
# MAGIC - Defines credentials for accessing the batch inference endpoint
# MAGIC - Defines the prompt that is used for batch inference

# COMMAND ----------

import httpx
import traceback
from mlflow.utils.databricks_utils import get_databricks_host_creds
from tenacity import (
    retry,
    stop_after_attempt,
    retry_if_exception,
    wait_random_exponential,
)


def is_backpressure(error: httpx.HTTPStatusError):
    if hasattr(error, "response") and hasattr(error.response, "status_code"):
        return error.response.status_code in (429, 503)


def is_other_error(error: httpx.HTTPStatusError):
    if hasattr(error, "response") and hasattr(error.response, "status_code"):
        return error.response.status_code != 503 and (
            error.response.status_code >= 500 or error.response.status_code == 408
        )


import re
import json


def extract_json_array(s: str) -> str:
    """
    Strips json array from the surrounding text
    :param s: string with json
    :return: string which contains just an array
    """
    groups = re.search(r"\{.*}", s, re.DOTALL)
    if groups:
        return groups.group()
    else:
        return s


def parse(s: str) -> str:
    """
    Tries parsing string into a json array
    :param s: string to parse
    :return: parsed list of questions
    """
    try:
        resp = json.loads(extract_json_array(s))
        if resp:
            return resp
        else:
            return None
    except Exception as e:
        return None


class AsyncChatClient:
    def __init__(self, chat_endpoint=True):
        self.client = httpx.AsyncClient(timeout=config_timeout)
        self.endpoint = config_endpoint
        self.prompt = config_prompt
        self.request_params = config_request_params
        self.chat_endpoint = chat_endpoint
        print(f"[AsyncChatClient] prompt: {self.prompt}")
        print(f"[AsyncChatClient] request parameters: {self.request_params}")

    @retry(
        retry=retry_if_exception(is_other_error),
        stop=stop_after_attempt(config_max_retries_other),
        wait=wait_random_exponential(multiplier=1, max=20),
    )
    @retry(
        retry=retry_if_exception(is_backpressure),
        stop=stop_after_attempt(config_max_retries_backpressure),
        wait=wait_random_exponential(multiplier=1, max=20),
    )
    async def predict(self, text):
        credencials = get_databricks_host_creds("databricks")
        url = f"{credencials.host}/serving-endpoints/{self.endpoint}/invocations"
        headers = {
            "Authorization": f"Bearer {credencials.token}",
            "Content-Type": "application/json",
        }

        messages = []
        if self.prompt:
            messages.append({"role": "user", "content": self.prompt + str(text)})
        else:
            messages.append({"role": "user", "content": str(text)})

        request_body = {"messages": messages, **self.request_params}
        if not self.chat_endpoint:
            request_body = {"prompt": messages[0]["content"], **self.request_params}

        response = await self.client.post(
            url=url,
            headers=headers,
            json=request_body,
        )
        response.raise_for_status()
        response = response.json()
        return (
            response["choices"][0]['text'],
            # response["choices"][0]["message"]["content"],
            response["usage"]["total_tokens"],
        )

    async def close(self):
        await self.client.aclose()

# COMMAND ----------

# MAGIC %md
# MAGIC The following processes the list of input data for batch inference that runs when this notebook is called from the **Perform batch inference on a provisioned throughput endpoint** .

# COMMAND ----------

# DBTITLE 1,Async Batch Inference Processor
import asyncio
import time
import tenacity
import httpx

class AsyncCounter:
    def __init__(self):
        self.value = 0

    async def increment(self):
        self.value += 1

async def generate(client, i, text_with_index, semaphore, counter, start_time):
    async with semaphore:
        try:
            index, text = text_with_index
            # content, content_parsed, num_tokens = await client.predict(text)
            content, num_tokens = await client.predict(text)
            response = (index, content, num_tokens, None)
        except Exception as e:
            print(f"{i}th request failed with exception: {e}")
            response = (index, None, None, 0, str(e))

        await counter.increment()
        if counter.value % config_logging_interval == 0:
            print(f"processed total {counter.value} requests in {time.time() - start_time:.2f} seconds.")
        return response

async def batch_inference(texts_with_index, chat_endpoint=True):
    semaphore = asyncio.Semaphore(config_concurrecy)
    counter = AsyncCounter()
    client = AsyncChatClient(chat_endpoint=chat_endpoint)
    start_time = time.time()

    tasks = [generate(client, i, text_with_index, semaphore, counter, start_time) for i, text_with_index in enumerate(texts_with_index)]
    responses = await asyncio.gather(*tasks)
    await client.close()

    return responses

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run batch inference
# MAGIC
# MAGIC The following code loads a Spark dataframe of the input data table and then converts that dataframe into a list of text that the model can process.

# COMMAND ----------

# DBTITLE 1,Dynamic Dataframe indexer
import uuid
from pyspark.sql.functions import monotonically_increasing_id

index_column = f"index_{uuid.uuid4().hex[:4]}"

## Step 1. Load spark dataframe from input table
source_sdf = spark.table(config_input_table).withColumn(index_column, monotonically_increasing_id())

input_sdf = source_sdf.select(index_column, config_input_column)
if config_input_num_rows:
    input_sdf = input_sdf.limit(config_input_num_rows)

## Step 2. Convert spark dataframe to list of texts
input_df = input_sdf.toPandas()
texts_with_index = [row for row in input_df.itertuples(index=False, name=None)]

# COMMAND ----------

# MAGIC %md
# MAGIC The following records and stores the batch inference responses.

# COMMAND ----------

# credencials = get_databricks_host_creds("databricks")
# url = f"{credencials.host}/serving-endpoints/{config_endpoint}/invocations"
# headers = {
#     "Authorization": f"Bearer {credencials.token}",
#     "Content-Type": "application/json",
# }

# messages = []
# messages.append({"prompt": str(texts_with_index[0][1])})
# import requests

# response = requests.post(
#     url=url,
#     headers=headers,
#     json={"prompt": str(texts_with_index[0][1]), **config_request_params},
# )
# response.raise_for_status()
# response = response.json()

# COMMAND ----------

# print(response["usage"]["total_tokens"])

# COMMAND ----------

# DBTITLE 1,Step 3: Batch Inference
start_time = time.time()
responses = await batch_inference(texts_with_index, chat_endpoint=False)
processing_time = time.time() - start_time
print(f"Total processing time: {processing_time:.2f} seconds.")

# COMMAND ----------

# MAGIC %md 
# MAGIC The following stores the output to a Unity Catalog table.

# COMMAND ----------

# DBTITLE 1,Step 4: Store Output to a Unity Catalog table
from pyspark.sql import functions as F
from pyspark.sql.functions import lit
from pyspark.sql.types import IntegerType, StructType, StructField, StringType

# Step 1: Create a DataFrame from model responses with indices
# schema = f"{index_column} long, resp_chat string, resp_chat_parsed struct<summary:string, sentiment:string, topic:string>, resp_total_tokens int, resp_error string"
schema = f"{index_column} long, resp_chat string, resp_total_tokens int, resp_error string"
response_sdf = spark.createDataFrame(responses, schema=schema) \
  .withColumn("resp_chat_parsed", F.from_json(F.col("resp_chat"), schema="struct<summary:string, sentiment:string, topic:string>"))


# Step 2: Join the DataFrames on the index column
output_sdf = source_sdf.join(response_sdf, on=index_column).drop(index_column)
output_sdf.display()
# Step 3: Persistent the final spark dataframe into UC output table
output_sdf.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(config_output_table)

# COMMAND ----------

# MAGIC %md Count the percent of nulls due to violated schema of the structured outputs

# COMMAND ----------

from pyspark.sql import functions as F
null_ct = output_sdf.filter(F.col("resp_chat_parsed").isNull()).count()
print(f"Total null ct: {null_ct}")
print(f"Percent null ct: {null_ct / config_input_num_rows}")

# COMMAND ----------

# MAGIC %md Convert the LLM output table to match the fine-tuning dataset format and write to delta table
# MAGIC
# MAGIC https://learn.microsoft.com/en-us/azure/databricks/large-language-models/foundation-model-training/data-preparation#prepare-data-for-supervised-fine-tuning

# COMMAND ----------

from pyspark.sql import functions as F

ft_dataset = spark.table(config_output_table).filter(F.col("resp_chat_parsed").isNotNull()) \
  .withColumn("response", F.to_json(F.col("resp_chat_parsed"))) \
  .select("prompt", "response")

display(ft_dataset)

# COMMAND ----------

config_output_table + "_ft_data"

# COMMAND ----------

ft_dataset.write.mode("overwrite").saveAsTable(config_output_table + "_ft_data")
