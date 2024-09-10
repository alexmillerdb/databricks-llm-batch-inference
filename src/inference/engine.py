from typing import List, Tuple, Optional, Dict, Any
import os
import sys

current_directory = os.getcwd()
root_directory = os.path.normpath(os.path.join(current_directory, '..', '..'))
sys.path.append(root_directory)

from src.api.client_interface import APIClientInterface

class InferenceEngine:
    def __init__(self, client: APIClientInterface):
        self.client = client

    async def infer(self, text: str) -> Tuple[str, int]:
        return await self.client.predict(text)