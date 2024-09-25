import pytest
from pyspark.sql import SparkSession
from pyspark.sql import Row
from llm_batch_inference.data.processor import DataProcessor
from llm_batch_inference.config.data_processor_config import DataProcessorConfig

from databricks.connect import DatabricksSession

from pyspark.sql import SparkSession


@pytest.fixture(scope="module")
def spark():
    return DatabricksSession.builder.getOrCreate()

@pytest.fixture
def config():
    return DataProcessorConfig(
        input_table_name="test_table",
        input_column_name="text",
        input_num_rows=2
    )

@pytest.fixture
def data_processor(spark, config):
    return DataProcessor(spark, config)

def test_load_spark_dataframe(spark, data_processor):
    # Create a test DataFrame
    data = [Row(text="text1"), Row(text="text2"), Row(text="text3")]
    df = spark.createDataFrame(data)
    df.createOrReplaceTempView("test_table")

    # Test load_spark_dataframe method
    result_df = data_processor.load_spark_dataframe()
    assert result_df is not None
    assert data_processor.index_column in result_df.columns

def test_select_and_limit_data(spark, data_processor):
    # Create a test DataFrame
    data = [Row(text="text1"), Row(text="text2"), Row(text="text3")]
    df = spark.createDataFrame(data)
    df.createOrReplaceTempView("test_table")

    # Load DataFrame and select/limit data
    source_df = data_processor.load_spark_dataframe()
    result_df = data_processor.select_and_limit_data(source_df)

    assert result_df is not None
    assert result_df.count() == 2  # As per the config input_num_rows
    assert data_processor.config.input_column_name in result_df.columns

def test_convert_to_list(spark, data_processor):
    # Create a test DataFrame
    data = [Row(text="text1"), Row(text="text2"), Row(text="text3")]
    df = spark.createDataFrame(data)
    df.createOrReplaceTempView("test_table")

    # Load DataFrame, select/limit data, and convert to list
    source_df = data_processor.load_spark_dataframe()
    input_df = data_processor.select_and_limit_data(source_df)
    result_list = data_processor.convert_to_list(input_df)

    assert result_list is not None
    assert len(result_list) == 2  # As per the config input_num_rows
    assert isinstance(result_list[0], tuple)

def test_process(spark, data_processor):
    # Create a test DataFrame
    data = [Row(text="text1"), Row(text="text2"), Row(text="text3")]
    df = spark.createDataFrame(data)
    df.createOrReplaceTempView("test_table")

    # Test the process method
    result_list = data_processor.process()

    assert result_list is not None
    assert len(result_list) == 2  # As per the config input_num_rows
    assert isinstance(result_list[0], tuple)

def test_get_source_sdf(spark, data_processor):
    # Create a test DataFrame
    data = [Row(text="text1"), Row(text="text2"), Row(text="text3")]
    df = spark.createDataFrame(data)
    df.createOrReplaceTempView("test_table")

    # Load DataFrame and test get_source_sdf method
    data_processor.load_spark_dataframe()
    source_df = data_processor.get_source_sdf()

    assert source_df is not None
    assert data_processor.index_column in source_df.columns

def test_get_input_sdf(spark, data_processor):
    # Create a test DataFrame
    data = [Row(text="text1"), Row(text="text2"), Row(text="text3")]
    df = spark.createDataFrame(data)
    df.createOrReplaceTempView("test_table")

    # Load DataFrame, select/limit data, and test get_input_sdf method
    data_processor.load_spark_dataframe()
    data_processor.select_and_limit_data(data_processor.get_source_sdf())
    input_df = data_processor.get_input_sdf()

    assert input_df is not None
    assert data_processor.config.input_column_name in input_df.columns

def test_get_texts_with_index(spark, data_processor):
    # Create a test DataFrame
    data = [Row(text="text1"), Row(text="text2"), Row(text="text3")]
    df = spark.createDataFrame(data)
    df.createOrReplaceTempView("test_table")

    # Load DataFrame, select/limit data, convert to list, and test get_texts_with_index method
    data_processor.process()
    texts_with_index = data_processor.get_texts_with_index()

    assert texts_with_index is not None
    assert len(texts_with_index) == 2  # As per the config input_num_rows
    assert isinstance(texts_with_index[0], tuple)