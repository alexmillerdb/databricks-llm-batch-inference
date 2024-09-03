# Databricks notebook source
# MAGIC %pip install -U --quiet mlflow mlflow[skinny] langchain langchain-community openai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog = "alex_m"
schema = "gen_ai"
endpoint = "llama38b"

# COMMAND ----------

from pyspark.sql import functions as F
from datasets import load_dataset

dataset = load_dataset("glnmario/news-qa-summarization")
train_dataset = dataset['train'].to_pandas()
spark_dataframe = spark.createDataFrame(train_dataset)
spark_dataframe.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.news_qa_summarization")

# COMMAND ----------

display(spark.table(f"{catalog}.{schema}.news_qa_summarization"))

# COMMAND ----------

# MAGIC %md ### Test structured outputs

# COMMAND ----------

prompt_template_str = """
  <|begin_of_text|><|start_header_id|>system<|end_header_id|>
  You are an AI assistant that specializes in data extraction. 
  Your task is to generate a structured response that helps answer the user query based on format instructions. 
  Please provide the output in JSON format as follows: 
  
  Below is an example of the structured output.
  Always format the output in JSON format as follows:

  ```json
  {{
    "summary": "summary of the story",
    "sentiment": "sentiment of the story; choices=["positive", "negative", "neutral"]",
    "topic": "topic of the story; choices=["finance", "technology", "sports", "politics", "crime", "weather"]"
  }}
  ```
  <|eot_id|><|start_header_id|>user<|end_header_id|>

  query: {query}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>
  """

# COMMAND ----------

from langchain_community.chat_models.databricks import ChatDatabricks
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(template=prompt_template_str, input_variables=["query"])
llm = ChatDatabricks(endpoint=endpoint, temperature=0.1)

chain = (prompt | llm | StrOutputParser()).with_retry(
    stop_after_attempt=3, wait_exponential_jitter=False
)
chain.invoke({"query": """Asuncion, Paraguay (CNN) -- Paraguayan President Fernando Lugo underwent prostate surgery early Friday, his spokesman said. The surgery, which was performed at the Italian Hospital in Asuncion, had been scheduled for weeks, spokesman Augusto dos Santos told reporters. The procedure was performed under local anesthesia, the spokesman said. Lugo's doctor, Nestor Martinez, said the operation was a transurethral resection, a surgery in which an instrument is inserted into the urethra to remove a section of the prostate that is blocking urine flow. Enlarged prostates are common among men as they get older. Lugo is 58. The surgery involved nine doctors and three nurses and took about an hour, Martinez said at a news conference. Lugo arrived at the hospital at 4 a.m. ( 2 a.m. ET), was wheeled into surgery at 5 a.m. and was in the recovery room by 6 a.m., Martinez said. The Paraguayan president's office released post-surgery photos of an alert-looking Lugo chatting with doctors and nurses while lying in a hospital bed. Lugo is a former Roman Catholic bishop who has been involved in several paternity controversies in the past year. He is expected to remain in the hospital until Saturday afternoon and then recuperate for three to four days in the presidential residence, spokesman dos Santos said. Lugo will carry on a restricted agenda while recuperating, the spokesman said Journalist Sanie Lopez Garelli contributed to this report."""})

# COMMAND ----------

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

# COMMAND ----------

df = spark.table(f"{catalog}.{schema}.news_qa_summarization").toPandas()
stories = df['story'].tolist()
stories

# COMMAND ----------

structured_outputs = []
story_list = []

# while len(structured_outputs) < 20:
for story in stories[:20]:
  response = parse(chain.invoke({"query": story}))
  if response:
    story_list.append(story)
    structured_outputs.append(response)

# COMMAND ----------

import pandas as pd
structured_df = pd.DataFrame({"story": story_list, "structured_outputs": structured_outputs})
display(structured_df)
# df = pd.DataFrame(structured_outputs).rename(columns={0:"question"})
# df = spark.createDataFrame(df)
# df.write.saveAsTable("rlaif.data.prompts_holdout")
# display(df)

# COMMAND ----------

# MAGIC %md ### Add prompt template to column to use in inference code

# COMMAND ----------

prompt_template_str = """
  <|begin_of_text|><|start_header_id|>system<|end_header_id|>
  You are an AI assistant that specializes in data extraction. 
  Your task is to generate a structured response that helps answer the user query based on format instructions. 
  Please provide the output in JSON format as follows: 
  
  Below is an example of the structured output.
  Always format the output in JSON format as follows:

  ```json
  {{
    "summary": "summary of the story",
    "sentiment": "sentiment of the story; choices=["positive", "negative", "neutral"]",
    "topic": "topic of the story; choices=["finance", "technology", "sports", "politics", "crime", "weather"]"
  }}
  ```
  <|eot_id|><|start_header_id|>user<|end_header_id|>

  query: {query}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>
  """

# COMMAND ----------

prompt_template_str = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
  
  You are an AI assistant that specializes in data extraction. When you receive a query, you should respond only with a structured response that helps answer the user query based on format instructions.
  Always provide the output in JSON format based on the users query and instructions.<|eot_id|>
  <|start_header_id|>user<|end_header_id|>
  
  Given the following query, respond with a JSON response that follows the format instructions.
  Below is an example of the structured output.

  ```json
  {{
    "summary": "summary of the story",
    "sentiment": "sentiment of the story; choices=["positive", "negative", "neutral"]",
    "topic": "topic of the story; choices=["finance", "technology", "sports", "politics", "crime", "weather"]"
  }}
  ```
  """
  # <|eot_id|><|start_header_id|>user<|end_header_id|>"""

  # query: {query}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>
  # """

# COMMAND ----------

from pyspark.sql import functions as F

df = spark.table(f"{catalog}.{schema}.news_qa_summarization") \
  .withColumn("prompt", F.concat(
    F.lit(prompt_template_str), 
    F.lit("\n query: "), F.col('story'), 
    F.lit("<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>")))

display(df)

# COMMAND ----------

df.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(f"{catalog}.{schema}.news_qa_summarization")

# COMMAND ----------

display(spark.table(f"{catalog}.{schema}.news_qa_summarization"))
