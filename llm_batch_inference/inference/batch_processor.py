from typing import List, Tuple, Optional, Any, Union
import asyncio
import time

from pyspark.sql import DataFrame
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, FloatType
from typing import Iterator
import pandas as pd

import nest_asyncio
from llm_batch_inference.inference.engine import InferenceEngine
from llm_batch_inference.config.inference_config import InferenceConfig
from llm_batch_inference.api.openai_client import OpenAIClient
from llm_batch_inference.utils.logger import Logger

class BatchProcessor:
    def __init__(self, engine: InferenceEngine, config: InferenceConfig):
        """
        Initializes the BatchProcessor with the given inference engine and configuration.

        Args:
            engine (InferenceEngine): The inference engine to be used for processing.
            config (InferenceConfig): The configuration settings for the inference process.
        """
        self.engine = engine
        self.config = config

    async def process_item_async(self, item: Tuple[int, str]) -> Tuple[int, Union[str, List[float]], int, Optional[str]]:
        """
        Asynchronously processes a single item using the inference engine.

        Args:
            item (Tuple[int, str]): A tuple containing an index and a text string to be processed.

        Returns:
            Tuple[int, Union[str, List[float]], int, Optional[str]]: A tuple containing:
                - index (int): The index of the processed item.
                - content (Union[str, List[float]]): The processed content, which could be a string or a list of floats.
                - num_tokens (int): The number of tokens processed.
                - error (Optional[str]): An error message if an exception occurred, otherwise None.
        """
        index, text = item
        try:
            content, num_tokens = await self.engine.async_infer(text)
            return (index, content, num_tokens, None)
        except Exception as e:
            return (index, None, 0, str(e))

    def process_item(self, item: Tuple[int, str]) -> Tuple[int, Union[str, List[float]], int, Optional[str]]:
        """
        Processes a single item using the inference engine.

        Args:
            item (Tuple[int, str]): A tuple containing an index and a text string.

        Returns:
            Tuple[int, Union[str, List[float]], int, Optional[str]]: A tuple containing:
                - index (int): The index of the item.
                - content (Union[str, List[float]]): The inferred content from the text, or None if an error occurred.
                - num_tokens (int): The number of tokens processed, or 0 if an error occurred.
                - error_message (Optional[str]): An error message if an exception was raised, otherwise None.
        """
        index, text = item
        try:
            content, num_tokens = self.engine.infer(text)
            return (index, content, num_tokens, None)
        except Exception as e:
            return (index, None, 0, str(e))

    async def process_batch_async(self, items: List[Tuple[int, str]]) -> List[Tuple[int, Union[str, List[float]], int, Optional[str]]]:
        """
        Asynchronously processes a batch of items with a concurrency limit.

        Args:
            items (List[Tuple[int, str]]): A list of tuples where each tuple contains an integer and a string.

        Returns:
            List[Tuple[int, Union[str, List[float]], int, Optional[str]]]: A list of tuples where each tuple contains:
                - An integer (same as the input integer).
                - A string or a list of floats (result of processing the input string).
                - An integer (status code or similar).
                - An optional string (error message or additional information).

        """
        semaphore = asyncio.Semaphore(self.config.concurrency)
        async def process_with_semaphore(item):
            async with semaphore:
                return await self.process_item_async(item)
        return await asyncio.gather(*[process_with_semaphore(item) for item in items])

    def process_batch(self, items: List[Tuple[int, str]]) -> List[Tuple[int, Union[str, List[float]], int, Optional[str]]]:
        """
        Processes a batch of items.

        Args:
            items (List[Tuple[int, str]]): A list of tuples where each tuple contains an integer and a string.

        Returns:
            List[Tuple[int, Union[str, List[float]], int, Optional[str]]]: A list of tuples where each tuple contains:
                - an integer
                - either a string or a list of floats
                - an integer
                - an optional string
        """
        return [self.process_item(item) for item in items]
    
class BatchInference:
    """
    BatchInference class for performing batch inference using OpenAI's API.
    Attributes:
        config (InferenceConfig): Configuration for inference.
        API_TOKEN (str): API token for authentication.
        API_ROOT (str): Root URL for the API.
        nest_asyncio_applied (bool): Flag to check if nest_asyncio has been applied.
        engine (InferenceEngine): Inference engine for processing.
        processor (BatchProcessor): Processor for handling batch inference.
        logger (Logger): Logger for tracking inference progress and time.
    Methods:
        _ensure_nest_asyncio():
            Ensures that nest_asyncio is applied to allow nested event loops.
        async __call__(texts_with_index: List[Tuple[int, str]]) -> List[Tuple[int, Union[str, List[float]], int, Optional[str]]]:
            Performs asynchronous batch inference on the provided texts.
        async run_batch_inference_async(texts_with_index: List[Tuple[int, str]]) -> List[Tuple[int, Union[str, List[float]], int, Optional[str]]]:
            Runs asynchronous batch inference.
        run_batch_inference(texts_with_index: List[Tuple[int, str]]) -> List[Tuple[int, Union[str, List[float]], int, Optional[str]]]:
            Runs batch inference synchronously.
        run_batch_inference_pandas_udf(df: DataFrame, input_col: str, output_cols: List[str], schema: StructType) -> DataFrame:
            Runs batch inference using a Pandas UDF for Spark DataFrame.
        get_output_schema() -> StructType:
            Returns the output schema based on the task type (embedding or not).
    """
    def __init__(self, config: InferenceConfig, API_TOKEN: str, API_ROOT: str):
        self.config = config
        self.API_TOKEN = API_TOKEN
        self.API_ROOT = API_ROOT
        self.nest_asyncio_applied = False
        client = OpenAIClient(config, API_ROOT=API_ROOT, API_TOKEN=API_TOKEN)
        self.engine = InferenceEngine(client)
        self.processor = BatchProcessor(self.engine, config)
        self.logger = Logger(config)

    def _ensure_nest_asyncio(self):
        """
        Ensures that `nest_asyncio` is applied to allow nested use of asyncio.

        This method checks if `nest_asyncio` has already been applied. If not, it attempts to apply it.
        If a `RuntimeError` is raised (indicating that `nest_asyncio` is already applied), the error is ignored.

        Attributes:
            nest_asyncio_applied (bool): A flag indicating whether `nest_asyncio` has been applied.
        """
        if not self.nest_asyncio_applied:
            try:
                nest_asyncio.apply()
                self.nest_asyncio_applied = True
            except RuntimeError:
                # nest_asyncio is already applied, ignore the error
                self.nest_asyncio_applied = True

    async def __call__(self, texts_with_index: List[Tuple[int, str]]) -> List[Tuple[int, Union[str, List[float]], int, Optional[str]]]:
        """
        Asynchronously processes a batch of texts with their corresponding indices.

        Args:
            texts_with_index (List[Tuple[int, str]]): A list of tuples where each tuple contains an index and a text string.

        Returns:
            List[Tuple[int, Union[str, List[float]], int, Optional[str]]]: A list of tuples where each tuple contains:
                - The original index.
                - The processed result which could be a string or a list of floats.
                - An integer (purpose not specified in the given code).
                - An optional string (purpose not specified in the given code).

        Logs:
            - The start time of the processing.
            - Progress for each processed item.
            - Total time taken to process the batch.
        """
        self.logger.start_time = time.time()  # Reset start time
        results = await self.processor.process_batch_async(texts_with_index)
        for _ in results:
            self.logger.log_progress()
        self.logger.log_total_time(len(texts_with_index))
        return results

    async def run_batch_inference_async(self, texts_with_index: List[Tuple[int, str]]) -> List[Tuple[int, Union[str, List[float]], int, Optional[str]]]:
        """
        Asynchronously runs batch inference on a list of texts with their corresponding indices.

        Args:
            texts_with_index (List[Tuple[int, str]]): A list of tuples where each tuple contains an index and a text string.

        Returns:
            List[Tuple[int, Union[str, List[float]], int, Optional[str]]]: A list of tuples where each tuple contains:
                - The original index.
                - The inference result, which can be either a string or a list of floats.
                - An integer (purpose not specified in the given code).
                - An optional string (purpose not specified in the given code).
        """
        return await self(texts_with_index)
    
    def run_batch_inference(self, texts_with_index: List[Tuple[int, str]]) -> List[Tuple[int, Union[str, List[float]], int, Optional[str]]]:
        """
        Runs batch inference on a list of texts with their corresponding indices.

        Args:
            texts_with_index (List[Tuple[int, str]]): A list of tuples where each tuple contains an index and a text string.

        Returns:
            List[Tuple[int, Union[str, List[float]], int, Optional[str]]]: A list of tuples where each tuple contains:
                - The original index.
                - The inference result, which can be a string or a list of floats.
                - The status code of the inference.
                - An optional error message if the inference failed.
        """
        self._ensure_nest_asyncio()
        return asyncio.run(self.run_batch_inference_async(texts_with_index))

    def run_batch_inference_pandas_udf(self, df: DataFrame, input_col: str, output_cols: List[str], schema: StructType) -> DataFrame:
        """
        Run batch inference using a Pandas UDF.
        This method applies a Pandas UDF to perform batch inference on a given DataFrame. It uses an inference engine to process
        text data and generate results, which are then appended to the DataFrame.
        Args:
            df (DataFrame): The input DataFrame containing the data to be processed.
            input_col (str): The name of the column in the DataFrame that contains the input text data.
            output_cols (List[str]): A list of column names for the output results. The list should contain three column names:
                - The first column will store the inference results.
                - The second column will store the number of tokens processed.
                - The third column will store any errors encountered during inference.
            schema (StructType): The schema of the output DataFrame.
        Returns:
            DataFrame: A new DataFrame with an additional column named "result" containing the inference results.
        """
        config = self.config
        api_token = self.API_TOKEN
        api_root = self.API_ROOT
        is_embedding = self.engine.is_embedding_task()

        @pandas_udf(schema)
        def inference_udf(iterator: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
            client = OpenAIClient(config, API_ROOT=api_root, API_TOKEN=api_token)
            engine = InferenceEngine(client)
            
            for batch in iterator:
                results = {col: [] for col in output_cols}
                for text in batch:
                    try:
                        content, num_tokens = engine.infer(text)
                        if is_embedding:
                            results[output_cols[0]].append(content)
                        else:
                            results[output_cols[0]].append(str(content))
                        results[output_cols[1]].append(num_tokens)
                        results[output_cols[2]].append(None)
                    except Exception as e:
                        results[output_cols[0]].append(None)
                        results[output_cols[1]].append(0)
                        results[output_cols[2]].append(str(e))

                yield pd.DataFrame(results)
                
        return df.withColumn("result", inference_udf(df[input_col]))

    def get_output_schema(self) -> StructType:
        """
        Generates the output schema for the batch processing results.
        Returns:
            StructType: A schema defining the structure of the output DataFrame.
                - If the task is an embedding task, the first field will be an array of floats.
                - Otherwise, the first field will be a string.
                - The second field is an integer.
                - The third field is a string.
        """
        if self.engine.is_embedding_task():
            content_field = StructField(self.config.output_column_names[0], ArrayType(FloatType()), True)
        else:
            content_field = StructField(self.config.output_column_names[0], StringType(), True)
        
        return StructType([
            content_field,
            StructField(self.config.output_column_names[1], IntegerType(), True),
            StructField(self.config.output_column_names[2], StringType(), True)
        ])
    
    # def process_with_pandas_udf(self, df: DataFrame, input_col: str, output_cols: List[str], schema: StructType) -> DataFrame:
    #     # Create a dynamic schema string based on output_cols and schema
    #     schema_string = ", ".join([f"{col} {schema[col].dataType.simpleString()}" for col in output_cols])
        
    #     @pandas_udf(schema_string)
    #     def chat_udf(iterator: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
    #         client = self.engine.client  # Assuming client is accessible
    #         for batch in iterator:
    #             results = {col: [] for col in output_cols}
    #             for text in batch:
    #                 try:
    #                     content, num_tokens = self.engine.infer(text)
    #                     results[output_cols[0]].append(content)
    #                     results[output_cols[1]].append(num_tokens)
    #                     results[output_cols[2]].append(None)
    #                 except Exception as e:
    #                     results[output_cols[0]].append(None)
    #                     results[output_cols[1]].append(0)
    #                     results[output_cols[2]].append(str(e))
    #             yield pd.DataFrame(results)

    #     result_df = df.withColumn("result", chat_udf(df[input_col]))
        
    #     # Dynamically select the output columns
    #     select_expr = ["*"] + [f"result.{col}" for col in output_cols]
    #     return result_df.select(*select_expr)
    
# class BatchInference:
#     def __init__(self, config: InferenceConfig, API_TOKEN: str, API_ROOT: str):
#         self.config = config
#         self.API_TOKEN = API_TOKEN
#         self.API_ROOT = API_ROOT
#         self.nest_asyncio_applied = False
#         client = OpenAIClient(config, API_ROOT=API_ROOT, API_TOKEN=API_TOKEN)
#         self.engine = InferenceEngine(client)
#         self.processor = BatchProcessor(self.engine, config)
#         self.logger = Logger(config)

#     def _ensure_nest_asyncio(self):
#         if not self.nest_asyncio_applied:
#             try:
#                 nest_asyncio.apply()
#                 self.nest_asyncio_applied = True
#             except RuntimeError:
#                 # nest_asyncio is already applied, ignore the error
#                 self.nest_asyncio_applied = True

#     async def __call__(self, texts_with_index: List[Tuple[int, str]]) -> List[Tuple[int, Optional[str], int, Optional[str]]]:
#         self.logger.start_time = time.time()  # Reset start time
#         results = await self.processor.process_batch_async(texts_with_index)
#         for _ in results:
#             self.logger.log_progress()
#         self.logger.log_total_time(len(texts_with_index))
#         return results

#     async def run_batch_inference_async(self, texts_with_index: List[Tuple[int, str]]) -> List[Tuple[int, Optional[str], int, Optional[str]]]:
#         return await self(texts_with_index)
    
#     def run_batch_inference(self, texts_with_index: List[Tuple[int, str]]) -> List[Tuple[int, Optional[str], int, Optional[str]]]:
#         self._ensure_nest_asyncio()
#         return asyncio.run(self.run_batch_inference_async(texts_with_index))

#     def run_batch_inference_pandas_udf(self, df: DataFrame, input_col: str, output_cols: List[str], schema: StructType) -> DataFrame:

#         config = self.config
#         api_token = self.API_TOKEN
#         api_root = self.API_ROOT

#         @pandas_udf(schema)
#         def chat_udf(iterator: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:

#             client = OpenAIClient(config, API_ROOT=api_root, API_TOKEN=api_token)
#             engine = InferenceEngine(client)
            
#             for batch in iterator:
#                 results = {col: [] for col in output_cols}
#                 for text in batch:
#                     try:
#                         content, num_tokens = engine.infer(text)
#                         results[output_cols[0]].append(content)
#                         results[output_cols[1]].append(num_tokens)
#                         results[output_cols[2]].append(None)
#                     except Exception as e:
#                         results[output_cols[0]].append(None)
#                         results[output_cols[1]].append(0)
#                         results[output_cols[2]].append(str(e))

#                 yield pd.DataFrame(results)
                
#         return df.withColumn("result", chat_udf(df[input_col]))