from typing import List, Tuple, Optional, Dict, Any, Union

from llm_batch_inference.api.client_interface import APIClientInterface
from llm_batch_inference.api.openai_client import OpenAIClient

class InferenceEngine:
    def __init__(self, client: OpenAIClient):
        self.client = client

    def infer(self, text: str) -> Union[Tuple[str, int], Tuple[List[float], int]]:
        """
        Perform inference on the given text.

        Args:
            text (str): The input text for inference.

        Returns:
            Union[Tuple[str, int], Tuple[List[float], int]]: 
                For chat and completion tasks: (generated_text, total_tokens)
                For embedding tasks: (embedding_vector, total_tokens)
        """
        return self.client.predict(text)

    async def async_infer(self, text: str) -> Union[Tuple[str, int], Tuple[List[float], int]]:
        """
        Perform asynchronous inference on the given text.

        Args:
            text (str): The input text for inference.

        Returns:
            Union[Tuple[str, int], Tuple[List[float], int]]: 
                For chat and completion tasks: (generated_text, total_tokens)
                For embedding tasks: (embedding_vector, total_tokens)
        """
        return await self.client.async_predict(text)

    def get_task_type(self) -> str:
        """
        Get the current task type (chat, completion, or embedding).

        Returns:
            str: The current task type.
        """
        return self.client.config.llm_task

    def is_embedding_task(self) -> bool:
        """
        Check if the current task is an embedding task.

        Returns:
            bool: True if the current task is embedding, False otherwise.
        """
        return self.get_task_type() == "embedding"