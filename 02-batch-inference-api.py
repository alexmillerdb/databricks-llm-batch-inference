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
config_logging_interval = 40

config_input_table = dbutils.widgets.get("input_table_name")
config_input_column = dbutils.widgets.get("input_column_name")
config_input_num_rows = dbutils.widgets.get("input_num_rows")
if config_input_num_rows:
    config_input_num_rows = int(config_input_num_rows)
config_output_table = dbutils.widgets.get("output_table_name")

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

# import mlflow
# import os
# import openai
# from openai import OpenAI

# # Get the API endpoint and token for the current notebook context
# API_ROOT = mlflow.utils.databricks_utils.get_databricks_host_creds().host
# API_TOKEN = mlflow.utils.databricks_utils.get_databricks_host_creds().token

# client = OpenAI(
#     api_key=API_TOKEN,
#     base_url=f"{API_ROOT}/serving-endpoints"
# )

# response = client.chat.completions.create(
#     model=config_endpoint,
#     messages=[
#       {
#         "role": "system",
#         "content": "You are a helpful assistant."
#       },
#       {
#         "role": "user",
#         "content": "What is a mixture of experts model?",
#       }
#     ],
#     max_tokens=256
# )
# prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
  
#   You are an AI assistant that specializes in data extraction. When you receive a query, you should respond only with a structured response that helps answer the user query based on format instructions.
#   Always provide the output in JSON format based on the users query and instructions.<|eot_id|>
#   <|start_header_id|>user<|end_header_id|>
  
#   Given the following query, respond with a JSON response that follows the format instructions.
#   Below is an example of the structured output.

#   ```json
#   {{
#     "summary": "summary of the story",
#     "sentiment": "sentiment of the story; choices=["positive", "negative", "neutral"]",
#     "topic": "topic of the story; choices=["finance", "technology", "sports", "politics", "crime", "weather"]"
#   }}
#   ```
  
#  query: ATLANTA, Georgia (CNN) -- They prefer the darkness and calm of early morning when their targets are most vulnerable, still sleeping or under the influence. They make sure their prey -- suspected killers and other violent fugitives -- know what they're up against. U.S. Marshal supervisory inspector James Ergas takes aim during a computer-simulated attack. "When they wake up to a submachine gun and flashlight in their face, they tend not to fight," says James Ergas, the supervisory inspector for the U.S. Marshals Southeast Regional Fugitive Task Force. The U.S. Marshals Service is the nation's oldest law enforcement agency and best known for protecting federal judges, transporting federal prisoners and protecting witnesses. Less known is the cutting-edge work of the agency's six regional task forces in capturing suspects. The task force in Atlanta is located in a nondescript warehouse office park. In 2007, the investigators from the Southeast task force arrested more than 3,000 suspects; only once did the Marshals exchange gunfire, Ergas says.  Watch Ergas blast bad guys in simulated attack » "This is the crème de la crème of the Marshal Service," says Eugene O'Donnell, a former prosecutor and New York City police officer who now teaches at the John Jay College of Criminal Justice in New York. On any given day, Ergas and his force are tracking 10 to 15 suspected killers roaming the Southeast, while also searching for other violent offenders. Already this year, they have been involved in a number of high-profile searches: Gary Michael Hilton, the suspect charged in the killing of Meredith Emerson who disappeared while hiking in northern Georgia; a fugitive Marine wanted in connection with the killing of Lance Cpl. Maria Lauterbach in North Carolina; and suspects wanted in connection with the killings of two suburban Atlanta police officers. But most of the time they're chasing suspects outside of the glare of the media spotlight. "Our mandate is to track violent fugitives -- murderers, armed robbers, rapists and fugitives of that caliber," says Keith Booker, the commander of the task force.  Watch Booker describe their mission » One suspect currently being hunted is Charles Leon Parker who has been on the run since the 1980s after being accused of molesting his stepdaughters. The Marshals were brought in recently, Booker says, after Parker allegedly called one of his victims and said, "I wanted you to know I saw you and your daughter, and she sure is beautiful." O'Donnell says it takes highly trained, high energy, "really special people" to do such work day in and day out, especially when they're up against "some of the most dangerous individuals in the country." "It's not an exaggeration to say they're the front of the front line," O'Donnell says. "It's not going to get any more challenging than this in law enforcement." To make sure they are well prepared, the Atlanta office is equipped with a locker full of high-powered weaponry; a high-tech operations center, complete with flat screen TVs, where they communicate directly with investigators in the field; a two-story house for training; and a 300-degree computer simulator that puts the Marshals into real life danger scenarios. In one demonstration, Ergas steps into the simulator and responds to reports of shots fired at a workplace. A woman rushes to a victim on the ground, as Ergas barks out commands. Moments later, a man rounds the corner. He too tends to the victim. Suddenly, the gunman runs into the corner and Ergas opens fire with his Glock. The suspect hits the ground.  Watch Ergas say there's no better training than the simulator » A split second later, another gunman emerges, and Ergas blasts him too. Think of it as Wii on steroids. "These are things you cannot get on a range," Ergas says. There are 50 different scenarios the simulator can create, with a technician able to change each scenario. A trainee can use a shotgun, rifle, Glock 22 or Glock 23. The guns shoot a laser<|eot_id|>
# <|start_header_id|>assistant<|end_header_id|>"""

# response = client.completions.create(
#     model="llama_8b_instruct_structured_outputs",
#     prompt=[prompt],
#     max_tokens=256
# )
# response


# COMMAND ----------

# Utility functions
def is_backpressure(error: httpx.HTTPStatusError):
    if hasattr(error, "response") and hasattr(error.response, "status_code"):
        return error.response.status_code in (429, 503)

def is_other_error(error: httpx.HTTPStatusError):
    if hasattr(error, "response") and hasattr(error.response, "status_code"):
        return error.response.status_code != 503 and (
            error.response.status_code >= 500 or error.response.status_code == 408
        )

class AsyncChatClient:
    def __init__(self, chat_endpoint=True):
        API_ROOT = mlflow.utils.databricks_utils.get_databricks_host_creds().host
        API_TOKEN = mlflow.utils.databricks_utils.get_databricks_host_creds().token

        self.client = OpenAI(
            api_key=API_TOKEN,
            base_url=f"{API_ROOT}/serving-endpoints"
        )
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
        # If the model is chat-based, use the ChatCompletion API
        if self.chat_endpoint:
            messages = [{"role": "user", "content": self.prompt + str(text) if self.prompt else str(text)}]
            try:
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.endpoint,
                    messages=messages,
                    **self.request_params
                )
                content = response.choices[0].message.content
                total_tokens = response.usage.total_tokens
                return content, total_tokens
            except Exception as e:
                print(f"Error while making OpenAI ChatCompletion API call: {e}")
                raise

        # If the model expects plain completion (non-chat)
        else:
            try:
                response = await asyncio.to_thread(
                    self.client.completions.create,
                    model=self.endpoint,
                    prompt=self.prompt + str(text) if self.prompt else str(text),
                    **self.request_params
                )
                content = response.choices[0].text
                total_tokens = response.usage.total_tokens
                return content, total_tokens
            except Exception as e:
                print(f"Error while making OpenAI Completion API call: {e}")
                raise

class AsyncCounter:
    def __init__(self):
        self.value = 0

    async def increment(self):
        self.value += 1

async def generate(client, i, text_with_index, semaphore, counter, start_time):
    async with semaphore:
        try:
            index, text = text_with_index
            content, num_tokens = await client.predict(text)
            response = (index, content, num_tokens, None)
        except Exception as e:
            print(f"{i}th request failed with exception: {e}")
            response = (index, None, 0, str(e))

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

    return responses


# COMMAND ----------

# MAGIC %md
# MAGIC ## Run batch inference
# MAGIC
# MAGIC The following code loads a Spark dataframe of the input data table and then converts that dataframe into a list of text that the model can process.

# COMMAND ----------

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

w = WorkspaceClient()

endpoint = w.serving_endpoints.get(config_endpoint)
endpoint_task = endpoint.task

if 'chat' in endpoint_task:
  chat_endpoint = True
elif 'completions' in endpoint_task:
  chat_endpoint = False

start_time = time.time()
responses = await batch_inference(texts_with_index, chat_endpoint=chat_endpoint)
processing_time = time.time() - start_time
print(f"Total processing time: {processing_time:.2f} seconds.")

# COMMAND ----------

# MAGIC %md 
# MAGIC The following stores the output to a Unity Catalog table.

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
response_sdf = spark.createDataFrame(responses, schema=schema) \
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
output_sdf.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(config_output_table)

# COMMAND ----------

null_ct = output_sdf.filter(F.col("resp_chat_parsed").isNull()).count()
print(f"Total null ct: {null_ct}")
print(f"Percent null ct: {null_ct / config_input_num_rows}")
