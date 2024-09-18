from typing import List, Tuple, Optional, Any
import asyncio
import time

from pyspark.sql import DataFrame
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from typing import Iterator
import pandas as pd

import nest_asyncio
from llm_batch_inference.inference.engine import InferenceEngine
from llm_batch_inference.config.inference_config import InferenceConfig
from llm_batch_inference.api.openai_client import OpenAIClient
from llm_batch_inference.utils.logger import Logger

class BatchProcessor:
    def __init__(self, engine: InferenceEngine, config: InferenceConfig):
        self.engine = engine
        self.config = config

    async def process_item_async(self, item: Tuple[int, str]) -> Tuple[int, Optional[str], int, Optional[str]]:
        index, text = item
        try:
            content, num_tokens = await self.engine.async_infer(text)
            return (index, content, num_tokens, None)
        except Exception as e:
            return (index, None, 0, str(e))

    def process_item(self, item: Tuple[int, str]) -> Tuple[int, Optional[str], int, Optional[str]]:
        index, text = item
        try:
            content, num_tokens = self.engine.infer(text)
            return (index, content, num_tokens, None)
        except Exception as e:
            return (index, None, 0, str(e))

    async def process_batch_async(self, items: List[Tuple[int, str]]) -> List[Tuple[int, Optional[str], int, Optional[str]]]:
        semaphore = asyncio.Semaphore(self.config.concurrency)
        async def process_with_semaphore(item):
            async with semaphore:
                return await self.process_item_async(item)
        return await asyncio.gather(*[process_with_semaphore(item) for item in items])

    def process_batch(self, items: List[Tuple[int, str]]) -> List[Tuple[int, Optional[str], int, Optional[str]]]:
        return [self.process_item(item) for item in items]
    
    def process_with_pandas_udf(self, df: DataFrame, input_col: str, output_cols: List[str], schema: StructType) -> DataFrame:
        # Create a dynamic schema string based on output_cols and schema
        schema_string = ", ".join([f"{col} {schema[col].dataType.simpleString()}" for col in output_cols])
        
        @pandas_udf(schema_string)
        def chat_udf(iterator: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
            client = self.engine.client  # Assuming client is accessible
            for batch in iterator:
                results = {col: [] for col in output_cols}
                for text in batch:
                    try:
                        content, num_tokens = self.engine.infer(text)
                        results[output_cols[0]].append(content)
                        results[output_cols[1]].append(num_tokens)
                        results[output_cols[2]].append(None)
                    except Exception as e:
                        results[output_cols[0]].append(None)
                        results[output_cols[1]].append(0)
                        results[output_cols[2]].append(str(e))
                yield pd.DataFrame(results)

        result_df = df.withColumn("result", chat_udf(df[input_col]))
        
        # Dynamically select the output columns
        select_expr = ["*"] + [f"result.{col}" for col in output_cols]
        return result_df.select(*select_expr)
    
class BatchInference:
    def __init__(self, config: InferenceConfig, API_TOKEN: str, API_ROOT: str):
        self.config = config
        nest_asyncio.apply()
        client = OpenAIClient(config, API_ROOT=API_ROOT, API_TOKEN=API_TOKEN)
        self.engine = InferenceEngine(client)
        self.processor = BatchProcessor(self.engine, config)
        self.logger = Logger(config)

    async def __call__(self, texts_with_index: List[Tuple[int, str]]) -> List[Tuple[int, Optional[str], int, Optional[str]]]:
        self.logger.start_time = time.time()  # Reset start time
        results = await self.processor.process_batch_async(texts_with_index)
        for _ in results:
            self.logger.log_progress()
        self.logger.log_total_time(len(texts_with_index))
        return results

    async def run_batch_inference_async(self, texts_with_index: List[Tuple[int, str]]) -> List[Tuple[int, Optional[str], int, Optional[str]]]:
        return await self(texts_with_index)

    def run_batch_inference(self, texts_with_index: List[Tuple[int, str]]) -> List[Tuple[int, Optional[str], int, Optional[str]]]:
        self.logger.start_time = time.time()  # Reset start time
        results = self.processor.process_batch(texts_with_index)
        for _ in results:
            self.logger.log_progress()
        self.logger.log_total_time(len(texts_with_index))
        return results

    def run_batch_inference_pandas_udf(self, df: DataFrame, input_col: str, output_cols: List[str], schema: StructType) -> DataFrame:

        result_df = self.processor.process_with_pandas_udf(df, input_col, output_cols, schema)

        return result_df