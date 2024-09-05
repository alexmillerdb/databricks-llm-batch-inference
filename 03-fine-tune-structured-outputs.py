# Databricks notebook source
# MAGIC %md ### Format data for chat completion or supervised fine-tuning: 
# MAGIC https://learn.microsoft.com/en-us/azure/databricks/large-language-models/foundation-model-training/data-preparation
# MAGIC
# MAGIC For chat completion task, chat-formatted data but be in a fine .jsonl format with below as chat-formatted data example
# MAGIC
# MAGIC `{"messages": [
# MAGIC   {"role": "system", "content": "A conversation between a user and a helpful assistant."},
# MAGIC   {"role": "user", "content": "Hi there. What's the capital of the moon?"},
# MAGIC   {"role": "assistant", "content": "This question doesn't make sense as nobody currently lives on the moon, meaning it would have no government or political institutions. Furthermore, international treaties prohibit any nation from asserting sovereignty over the moon and other celestial bodies."},
# MAGIC   ]
# MAGIC }`
# MAGIC
# MAGIC Preparation example for supervised fine-tuning
# MAGIC
# MAGIC `{"prompt": "your-custom-prompt", "response": "your-custom-response"}`

# COMMAND ----------

# MAGIC %pip install databricks-genai databricks-sdk --upgrade
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

ft_table_path = "alex_m.gen_ai.news_qa_summarization_llm_output_ft_data"
MODEL = 'meta-llama/Meta-Llama-3-8B-Instruct' # "student" model
# MODEL = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
REGISTER_TO = 'alex_m.gen_ai.llama_8b_instruct_structured_outputs' # where to register your finetuned model for deployment
DATA_PREP_CLUSTER_ID = spark.conf.get("spark.databricks.clusterUsageTags.clusterId") # spark cluster to prepare your UC table for training

# COMMAND ----------

from pyspark.sql import functions as F

data = spark.table(ft_table_path)

print(f"Dataset count: {data.count()}")
display(data)

# COMMAND ----------

from databricks.model_training import foundation_model as fm

finetuning_run = fm.create(
    model=MODEL,
    train_data_path=ft_table_path,
    data_prep_cluster_id=DATA_PREP_CLUSTER_ID,
    register_to=REGISTER_TO,
    task_type="INSTRUCTION_FINETUNE",
    training_duration="5ep"
)

# COMMAND ----------

finetuning_run.get_events()

# COMMAND ----------

import mlflow
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
def get_latest_model_version(model_name):
    mlflow_client = MlflowClient()
    latest_version = 1
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

# COMMAND ----------

get_latest_model_version(REGISTER_TO)

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput, EndpointCoreConfigInput, AutoCaptureConfigInput

serving_endpoint_name = "llama_8b_instruct_structured_outputs"
catalog = "alex_m"
db = "gen_ai"

w = WorkspaceClient()
endpoint_config = EndpointCoreConfigInput(
    name=serving_endpoint_name,
    served_entities=[
        ServedEntityInput(
            entity_name=REGISTER_TO,
            entity_version=get_latest_model_version(REGISTER_TO),
            min_provisioned_throughput=0, # The minimum tokens per second that the endpoint can scale down to.
            max_provisioned_throughput=3600,# The maximum tokens per second that the endpoint can scale up to.
            scale_to_zero_enabled=True
        )
    ],
    auto_capture_config = AutoCaptureConfigInput(catalog_name=catalog, schema_name=db, enabled=True, table_name_prefix="fine_tuned_llm_frontier")
)

force_update = False #Set this to True to release a newer version (the demo won't update the endpoint to a newer model version by default)
existing_endpoint = next(
    (e for e in w.serving_endpoints.list() if e.name == serving_endpoint_name), None
)
if existing_endpoint == None:
    print(f"Creating the endpoint {serving_endpoint_name}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.create_and_wait(name=serving_endpoint_name, config=endpoint_config)
else:
  print(f"endpoint {serving_endpoint_name} already exist...")
  if force_update:
    w.serving_endpoints.update_config_and_wait(served_entities=endpoint_config.served_entities, name=serving_endpoint_name)
