import pytest
from pydantic import ValidationError
from llm_batch_inference.config.data_processor_config import DataProcessorConfig

def test_valid_config():
    config = DataProcessorConfig(
        input_table_name="test_table",
        input_column_name="test_column",
        input_num_rows=100
    )
    assert config.input_table_name == "test_table"
    assert config.input_column_name == "test_column"
    assert config.input_num_rows == 100

def test_optional_num_rows():
    config = DataProcessorConfig(
        input_table_name="test_table",
        input_column_name="test_column"
    )
    assert config.input_table_name == "test_table"
    assert config.input_column_name == "test_column"
    assert config.input_num_rows is None

def test_missing_required_fields():
    with pytest.raises(ValidationError):
        DataProcessorConfig()

def test_invalid_extra_fields():
    with pytest.raises(ValidationError):
        DataProcessorConfig(
            input_table_name="test_table",
            input_column_name="test_column",
            input_num_rows=100,
            extra_field="not_allowed"
        )