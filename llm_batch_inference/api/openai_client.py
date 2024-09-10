from typing import List, Tuple, Optional, Dict, Any
import asyncio
import mlflow
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
    async def predict(self, text: str) -> Tuple[str, int]:
        # If the model is chat-based, use the ChatCompletion API
        if self.config.llm_task == "chat":
            messages = [{"role": "user", "content": self.config.prompt + str(text) if self.config.prompt else str(text)}]
            try:
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.config.endpoint,
                    messages=messages,
                    **self.config.request_params
                )
                content = response.choices[0].message.content
                total_tokens = response.usage.total_tokens
                return content, total_tokens
            except Exception as e:
                print(f"Error while making OpenAI ChatCompletion API call: {e}")
                raise

        # If the model expects plain completion (non-chat)
        elif self.config.llm_task == "completion":
            try:
                response = await asyncio.to_thread(
                    self.client.completions.create,
                    model=self.config.endpoint,
                    prompt=self.config.prompt + str(text) if self.config.prompt else str(text),
                    **self.config.request_params
                )
                content = response.choices[0].text
                total_tokens = response.usage.total_tokens
                return content, total_tokens
            except Exception as e:
                print(f"Error while making OpenAI Completion API call: {e}")
                raise