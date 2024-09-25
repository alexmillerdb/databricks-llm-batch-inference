from typing import List, Tuple, Optional, Dict, Any, Union
import asyncio
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception

from llm_batch_inference.api.client_interface import APIClientInterface
from llm_batch_inference.config.inference_config import InferenceConfig
from llm_batch_inference.utils.error_handlers import is_backpressure, is_other_error


class OpenAIClient(APIClientInterface):
    def __init__(self, config: InferenceConfig, API_ROOT: str, API_TOKEN: str):
        self.config = config
        self.client = OpenAI(
            api_key=API_TOKEN,
            base_url=f"{API_ROOT}/serving-endpoints"
        )

    @retry(
        retry=retry_if_exception(is_other_error),
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1, max=20),
    )
    @retry(
        retry=retry_if_exception(is_backpressure),
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1, max=20),
    )
    async def async_predict(self, text: str) -> Union[Tuple[str, int], Tuple[List[float], int]]:
        try:
            if self.config.llm_task == "chat":
                return await self._async_chat_completion(text)
            elif self.config.llm_task == "completion":
                return await self._async_completion(text)
            elif self.config.llm_task == "embedding":
                return await self._async_embedding(text)
            else:
                raise ValueError(f"Unsupported llm_task: {self.config.llm_task}")
        except Exception as e:
            print(f"Error in async_predict: {e}")
            raise

    def predict(self, text: str) -> Union[Tuple[str, int], Tuple[List[float], int]]:
        try:
            if self.config.llm_task == "chat":
                return self._chat_completion(text)
            elif self.config.llm_task == "completion":
                return self._completion(text)
            elif self.config.llm_task == "embedding":
                return self._embedding(text)
            else:
                raise ValueError(f"Unsupported llm_task: {self.config.llm_task}")
        except Exception as e:
            print(f"Error in predict: {e}")
            raise

    async def _async_chat_completion(self, text: str) -> Tuple[str, int]:
        messages = [{"role": "user", "content": self.config.prompt + str(text) if self.config.prompt else str(text)}]
        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.config.endpoint,
            messages=messages,
            **self.config.request_params
        )
        content = response.choices[0].message.content if response.choices and response.choices[0].message else None
        total_tokens = response.usage.total_tokens if response.usage else 0
        if content is None:
            raise ValueError("Received empty content from OpenAI ChatCompletion API")
        return content, total_tokens

    async def _async_completion(self, text: str) -> Tuple[str, int]:
        response = await asyncio.to_thread(
            self.client.completions.create,
            model=self.config.endpoint,
            prompt=self.config.prompt + str(text) if self.config.prompt else str(text),
            **self.config.request_params
        )
        content = response.choices[0].text if response.choices else None
        total_tokens = response.usage.total_tokens if response.usage else 0
        if content is None:
            raise ValueError("Received empty content from OpenAI Completion API")
        return content, total_tokens

    async def _async_embedding(self, text: str) -> Tuple[List[float], int]:
        response = await asyncio.to_thread(
            self.client.embeddings.create,
            input=text,
            model=self.config.endpoint
        )
        content = response.data[0].embedding
        total_tokens = response.usage.total_tokens
        return content, total_tokens

    def _chat_completion(self, text: str) -> Tuple[str, int]:
        messages = [{"role": "user", "content": self.config.prompt + str(text) if self.config.prompt else str(text)}]
        response = self.client.chat.completions.create(
            model=self.config.endpoint,
            messages=messages,
            **self.config.request_params
        )
        content = response.choices[0].message.content
        total_tokens = response.usage.total_tokens
        return content, total_tokens

    def _completion(self, text: str) -> Tuple[str, int]:
        response = self.client.completions.create(
            model=self.config.endpoint,
            prompt=self.config.prompt + str(text) if self.config.prompt else str(text),
            **self.config.request_params
        )
        content = response.choices[0].text
        total_tokens = response.usage.total_tokens
        return content, total_tokens

    def _embedding(self, text: str) -> Tuple[List[float], int]:
        response = self.client.embeddings.create(
            input=text,
            model=self.config.endpoint
        )
        content = response.data[0].embedding
        total_tokens = response.usage.total_tokens
        return content, total_tokens