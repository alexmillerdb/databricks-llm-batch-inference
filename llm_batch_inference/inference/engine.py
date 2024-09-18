from typing import List, Tuple, Optional, Dict, Any

from llm_batch_inference.api.client_interface import APIClientInterface
from llm_batch_inference.api.openai_client import OpenAIClient

class InferenceEngine:
    def __init__(self, client: OpenAIClient):
        self.client = client

    def infer(self, text: str) -> Tuple[str, int]:
        return self.client.predict(text)

    async def async_infer(self, text: str) -> Tuple[str, int]:
        return await self.client.async_predict(text)

# class InferenceEngine:
#     def __init__(self, client: APIClientInterface):
#         self.client = client

#     async def infer(self, text: str) -> Tuple[str, int]:
#         return await self.client.predict(text)