from pydantic import BaseModel, Field
from typing import List, Tuple, Optional, Dict, Any, Literal

class InferenceConfig(BaseModel):
    """
    InferenceConfig is a configuration class for setting up inference parameters.

    Attributes:
        endpoint (str): The endpoint for the inference service. Defaults to "databricks-meta-llama-3-1-70b-instruct".
        timeout (int): The timeout duration for inference requests in seconds. Defaults to 300.
        max_retries_backpressure (int): The maximum number of retries for backpressure errors. Defaults to 3.
        max_retries_other (int): The maximum number of retries for other errors. Defaults to 3.
        prompt (Optional[str]): The prompt to be used for inference. Defaults to None.
        request_params (Dict): Additional parameters for the inference request. Defaults to an empty dictionary.
        concurrency (int): The number of concurrent inference requests allowed. Defaults to 15.
        logging_interval (int): The interval for logging in seconds. Defaults to 40.
        enable_logging (bool): Flag to enable or disable logging. Defaults to True.
        llm_task (Literal["chat", "completion", "embedding"]): The task type for the language model. Choices are "chat", "completion", and "embedding". Defaults to an empty string.

        model_config (dict): Configuration for the model, with "extra" set to "forbid".
    """
    endpoint: str = Field(default="databricks-meta-llama-3-1-70b-instruct")
    timeout: int = Field(default=300, ge=1)
    max_retries_backpressure: int = Field(default=3)
    max_retries_other: int = Field(default=3)
    prompt: Optional[str] = Field(default=None)
    request_params: Dict = Field(default_factory=dict)
    concurrency: int = Field(default=15, ge=1)
    logging_interval: int = Field(default=40)
    enable_logging: bool = Field(default=False)
    llm_task: Literal["chat", "completion", "embedding"] = Field(default="")

    model_config = {
        "extra": "forbid"
    }