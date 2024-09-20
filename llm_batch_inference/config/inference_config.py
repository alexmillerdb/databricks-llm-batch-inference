from pydantic import BaseModel, Field
from typing import List, Tuple, Optional, Dict, Any

class InferenceConfig(BaseModel):
    endpoint: str = Field(default="databricks-meta-llama-3-1-70b-instruct")
    timeout: int = Field(default=300)
    max_retries_backpressure: int = Field(default=3)
    max_retries_other: int = Field(default=3)
    prompt: Optional[str] = Field(default=None)
    request_params: Dict = Field(default_factory=dict)
    concurrency: int = Field(default=15)
    logging_interval: int = Field(default=40)
    enable_logging: bool = Field(default=True)
    llm_task: str = Field(
        default="",
        json_schema_extra={
            "choices": ["chat", "completion", "embedding"]
        }
    )

    model_config = {
        "extra": "forbid"
    }