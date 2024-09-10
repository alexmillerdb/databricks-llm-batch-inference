from typing import List, Tuple, Optional, Dict, Any
import asyncio
import time

from inference.engine import InferenceEngine
from config.inference_config import InferenceConfig
from api.openai_client import OpenAIClient
from utils.logger import Logger

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
    
class BatchInference:
    def __init__(self, config: InferenceConfig, API_TOKEN: str, API_ROOT: str):
        self.config = config
        client = OpenAIClient(config, API_ROOT=API_ROOT, API_TOKEN=API_TOKEN)
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
    
    async def run_batch_inference(self, texts_with_index: List[Tuple[int, str]]) -> List[Tuple[int, Optional[str], int, Optional[str]]]:
        return await self(texts_with_index)