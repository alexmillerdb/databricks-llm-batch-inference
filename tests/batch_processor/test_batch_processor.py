import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import Tuple, Optional, List

from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql import DataFrame

from llm_batch_inference.inference.engine import InferenceEngine
from llm_batch_inference.config.inference_config import InferenceConfig
from llm_batch_inference.inference.batch_processor import BatchProcessor, BatchInference
from llm_batch_inference.api.openai_client import OpenAIClient
from llm_batch_inference.utils.logger import Logger

@pytest.fixture
def processor():
    mock_engine = MagicMock(spec=InferenceEngine)
    mock_config = MagicMock(spec=InferenceConfig)
    mock_config.concurrency = 2
    return BatchProcessor(mock_engine, mock_config)

@pytest.fixture
def batch_inference():
    mock_config = MagicMock(spec=InferenceConfig)
    mock_config.enable_logging = True
    mock_config.concurrency = 2
    mock_client = MagicMock(spec=OpenAIClient)
    mock_engine = MagicMock(spec=InferenceEngine)
    mock_processor = MagicMock(spec=BatchProcessor)
    mock_logger = MagicMock(spec=Logger)
    
    mock_engine.client = mock_client
    mock_processor.process_batch.return_value = [(1, "processed text", 5, None)]
    mock_processor.process_batch_async = AsyncMock(return_value=[(1, "processed text", 5, None)])
    
    bi = BatchInference(mock_config, "fake_token", "fake_root")
    bi.engine = mock_engine
    bi.processor = mock_processor
    bi.logger = mock_logger
    return bi

def test_process_item(processor):
    item = (1, "test text")
    processor.engine.infer.return_value = ("processed text", 5)
    
    result = processor.process_item(item)
    
    assert result == (1, "processed text", 5, None)
    processor.engine.infer.assert_called_once_with("test text")

def test_process_item_exception(processor):
    item = (1, "test text")
    processor.engine.infer.side_effect = Exception("Inference error")
    
    result = processor.process_item(item)
    
    assert result == (1, None, 0, "Inference error")
    processor.engine.infer.assert_called_once_with("test text")

@pytest.mark.asyncio
async def test_process_item_async(processor):
    item = (1, "test text")
    processor.engine.async_infer = AsyncMock(return_value=("processed text", 5))
    
    result = await processor.process_item_async(item)
    
    assert result == (1, "processed text", 5, None)
    processor.engine.async_infer.assert_called_once_with("test text")

@pytest.mark.asyncio
async def test_process_item_async_exception(processor):
    item = (1, "test text")
    processor.engine.async_infer = AsyncMock(side_effect=Exception("Inference error"))
    
    result = await processor.process_item_async(item)
    
    assert result == (1, None, 0, "Inference error")
    processor.engine.async_infer.assert_called_once_with("test text")

def test_process_batch(processor):
    items = [(1, "text1"), (2, "text2")]
    processor.engine.infer.side_effect = [("processed text1", 5), ("processed text2", 6)]
    
    result = processor.process_batch(items)
    
    expected = [(1, "processed text1", 5, None), (2, "processed text2", 6, None)]
    assert result == expected

@pytest.mark.asyncio
async def test_process_batch_async(processor):
    items = [(1, "text1"), (2, "text2")]
    processor.engine.async_infer = AsyncMock(side_effect=[("processed text1", 5), ("processed text2", 6)])
    
    result = await processor.process_batch_async(items)
    
    expected = [(1, "processed text1", 5, None), (2, "processed text2", 6, None)]
    assert result == expected

def test_run_batch_inference(batch_inference):
    texts_with_index = [(1, "test text")]
    
    result = batch_inference.run_batch_inference(texts_with_index)
    
    assert result == [(1, "processed text", 5, None)]
    batch_inference.processor.process_batch.assert_called_once_with(texts_with_index)
    batch_inference.logger.log_progress.assert_called()
    batch_inference.logger.log_total_time.assert_called_once_with(len(texts_with_index))

@pytest.mark.asyncio
async def test_run_batch_inference_async(batch_inference):
    texts_with_index = [(1, "test text")]
    
    result = await batch_inference.run_batch_inference_async(texts_with_index)
    
    assert result == [(1, "processed text", 5, None)]
    batch_inference.processor.process_batch_async.assert_called_once_with(texts_with_index)
    batch_inference.logger.log_progress.assert_called()
    batch_inference.logger.log_total_time.assert_called_once_with(len(texts_with_index))

def test_run_batch_inference_pandas_udf(batch_inference):
    df = MagicMock(spec=DataFrame)
    input_col = "input"
    output_cols = ["output1", "output2", "output3"]
    schema = StructType([
        StructField("output1", StringType(), True),
        StructField("output2", IntegerType(), True),
        StructField("output3", StringType(), True)
    ])
    
    result_df = batch_inference.run_batch_inference_pandas_udf(df, input_col, output_cols, schema)
    
    batch_inference.processor.process_with_pandas_udf.assert_called_once_with(df, input_col, output_cols, schema)
    assert result_df == batch_inference.processor.process_with_pandas_udf.return_value

# import unittest
# from unittest.mock import AsyncMock, MagicMock
# from typing import Tuple, Optional, List

# from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# print(sys.path)

# from llm_batch_inference.inference.engine import InferenceEngine
# from llm_batch_inference.config.inference_config import InferenceConfig
# from llm_batch_inference.inference.batch_processor import BatchProcessor, BatchInference
# from llm_batch_inference.api.openai_client import OpenAIClient
# from llm_batch_inference.utils.logger import Logger

# class TestBatchProcessor(unittest.TestCase):
#     def setUp(self):
#         self.mock_engine = MagicMock(spec=InferenceEngine)
#         self.mock_config = MagicMock(spec=InferenceConfig)
#         self.mock_config.concurrency = 2
#         self.processor = BatchProcessor(self.mock_engine, self.mock_config)

#     def test_process_item(self):
#         item = (1, "test text")
#         self.mock_engine.infer.return_value = ("processed text", 5)
        
#         result = self.processor.process_item(item)
        
#         self.assertEqual(result, (1, "processed text", 5, None))
#         self.mock_engine.infer.assert_called_once_with("test text")

#     def test_process_item_exception(self):
#         item = (1, "test text")
#         self.mock_engine.infer.side_effect = Exception("Inference error")
        
#         result = self.processor.process_item(item)
        
#         self.assertEqual(result, (1, None, 0, "Inference error"))
#         self.mock_engine.infer.assert_called_once_with("test text")

#     async def test_process_item_async(self):
#         item = (1, "test text")
#         self.mock_engine.async_infer = AsyncMock(return_value=("processed text", 5))
        
#         result = await self.processor.process_item_async(item)
        
#         self.assertEqual(result, (1, "processed text", 5, None))
#         self.mock_engine.async_infer.assert_called_once_with("test text")

#     async def test_process_item_async_exception(self):
#         item = (1, "test text")
#         self.mock_engine.async_infer = AsyncMock(side_effect=Exception("Inference error"))
        
#         result = await self.processor.process_item_async(item)
        
#         self.assertEqual(result, (1, None, 0, "Inference error"))
#         self.mock_engine.async_infer.assert_called_once_with("test text")

#     def test_process_batch(self):
#         items = [(1, "text1"), (2, "text2")]
#         self.mock_engine.infer.side_effect = [("processed text1", 5), ("processed text2", 6)]
        
#         result = self.processor.process_batch(items)
        
#         expected = [(1, "processed text1", 5, None), (2, "processed text2", 6, None)]
#         self.assertEqual(result, expected)

#     async def test_process_batch_async(self):
#         items = [(1, "text1"), (2, "text2")]
#         self.mock_engine.async_infer = AsyncMock(side_effect=[("processed text1", 5), ("processed text2", 6)])
        
#         result = await self.processor.process_batch_async(items)
        
#         expected = [(1, "processed text1", 5, None), (2, "processed text2", 6, None)]
#         self.assertEqual(result, expected)

# class TestBatchInference(unittest.TestCase):
#     def setUp(self):
#         self.mock_config = MagicMock(spec=InferenceConfig)
#         self.mock_client = MagicMock(spec=OpenAIClient)
#         self.mock_engine = MagicMock(spec=InferenceEngine)
#         self.mock_processor = MagicMock(spec=BatchProcessor)
#         self.mock_logger = MagicMock(spec=Logger)
        
#         self.mock_config.concurrency = 2
#         self.mock_engine.client = self.mock_client
#         self.mock_processor.process_batch.return_value = [(1, "processed text", 5, None)]
#         self.mock_processor.process_batch_async = AsyncMock(return_value=[(1, "processed text", 5, None)])
        
#         self.batch_inference = BatchInference(self.mock_config, "fake_token", "fake_root")
#         self.batch_inference.engine = self.mock_engine
#         self.batch_inference.processor = self.mock_processor
#         self.batch_inference.logger = self.mock_logger

#     def test_run_batch_inference(self):
#         texts_with_index = [(1, "test text")]
        
#         result = self.batch_inference.run_batch_inference(texts_with_index)
        
#         self.assertEqual(result, [(1, "processed text", 5, None)])
#         self.mock_processor.process_batch.assert_called_once_with(texts_with_index)
#         self.mock_logger.log_progress.assert_called()
#         self.mock_logger.log_total_time.assert_called_once_with(len(texts_with_index))

#     async def test_run_batch_inference_async(self):
#         texts_with_index = [(1, "test text")]
        
#         result = await self.batch_inference.run_batch_inference_async(texts_with_index)
        
#         self.assertEqual(result, [(1, "processed text", 5, None)])
#         self.mock_processor.process_batch_async.assert_called_once_with(texts_with_index)
#         self.mock_logger.log_progress.assert_called()
#         self.mock_logger.log_total_time.assert_called_once_with(len(texts_with_index))

#     def test_run_batch_inference_pandas_udf(self):
#         from pyspark.sql import DataFrame
        
#         df = MagicMock(spec=DataFrame)
#         input_col = "input"
#         output_cols = ["output1", "output2", "output3"]
#         schema = StructType([
#             StructField("output1", StringType(), True),
#             StructField("output2", IntegerType(), True),
#             StructField("output3", StringType(), True)
#         ])
        
#         result_df = self.batch_inference.run_batch_inference_pandas_udf(df, input_col, output_cols, schema)
        
#         self.mock_processor.process_with_pandas_udf.assert_called_once_with(df, input_col, output_cols, schema)
#         self.assertEqual(result_df, self.mock_processor.process_with_pandas_udf.return_value)


# if __name__ == '__main__':
#     unittest.main()