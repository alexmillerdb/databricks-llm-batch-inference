from pydantic import BaseModel, Field
from typing import Optional

class DataProcessorConfig(BaseModel):
    input_table_name: str = Field(..., description="Name of the input table")
    input_column_name: str = Field(..., description="Name of the input column")
    input_num_rows: Optional[int] = Field(None, description="Number of rows to process (optional)")

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
