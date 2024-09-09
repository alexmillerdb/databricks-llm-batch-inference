from typing import List, Tuple, Optional, Dict, Any
from api.client_interface import APIClientInterface

class InferenceEngine:
    def __init__(self, client: APIClientInterface):
        self.client = client

    async def infer(self, text: str) -> Tuple[str, int]:
        return await self.client.predict(text)