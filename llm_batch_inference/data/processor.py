import uuid
from typing import List, Tuple, Optional
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import monotonically_increasing_id
import pandas as pd

from llm_batch_inference.config.data_processor_config import DataProcessorConfig

class DataProcessor:
    """
    DataProcessor is a class responsible for processing data using Spark. It loads data from a specified input table,
    selects and limits the data based on configuration, and converts the data into a list of tuples.

    Attributes:
        spark (SparkSession): The Spark session used for data processing.
        config (DataProcessorConfig): Configuration object containing input table name, input column name, and optional row limit.
        index_column (str): A unique index column name generated for the DataFrame.
        source_sdf (Optional[DataFrame]): The source Spark DataFrame loaded from the input table.
        input_sdf (Optional[DataFrame]): The Spark DataFrame after selecting and limiting the data.
        texts_with_index (Optional[List[Tuple[int, str]]]): The processed list of tuples containing index and text.

    Methods:
        load_spark_dataframe() -> DataFrame:
            Load the Spark DataFrame from the input table and add an index column.

        select_and_limit_data(sdf: DataFrame) -> DataFrame:
            Select required columns and optionally limit the number of rows.

        convert_to_list(sdf: DataFrame) -> List[Tuple[int, str]]:
            Convert Spark DataFrame to a list of tuples.

        process() -> List[Tuple[int, str]]:
            Main method to process the data.

        get_source_sdf() -> Optional[DataFrame]:
            Get the source Spark DataFrame.

        get_input_sdf() -> Optional[DataFrame]:
            Get the input Spark DataFrame.

        get_texts_with_index() -> Optional[List[Tuple[int, str]]]:
            Get the processed list of texts with index.
    """
    def __init__(self, spark: SparkSession, config: DataProcessorConfig):
        """
        Initializes the DataProcessor with a Spark session and configuration.

        Args:
            spark (SparkSession): The Spark session to use for data processing.
            config (DataProcessorConfig): Configuration settings for the data processor.

        Attributes:
            spark (SparkSession): The Spark session to use for data processing.
            config (DataProcessorConfig): Configuration settings for the data processor.
            index_column (str): A unique index column name generated using a UUID.
            source_sdf (Optional[DataFrame]): The source Spark DataFrame, initially set to None.
            input_sdf (Optional[DataFrame]): The input Spark DataFrame, initially set to None.
            texts_with_index (Optional[List[Tuple[int, str]]]): A list of tuples containing index and text, initially set to None.
        """
        self.spark = spark
        self.config = config
        self.index_column = f"index_{uuid.uuid4().hex[:4]}"
        self.source_sdf: Optional[DataFrame] = None
        self.input_sdf: Optional[DataFrame] = None
        self.texts_with_index: Optional[List[Tuple[int, str]]] = None

    def load_spark_dataframe(self) -> DataFrame:
        """
        Load the Spark DataFrame from the input table and add an index column.

        This method reads a table specified by the configuration's input table name
        into a Spark DataFrame and adds a monotonically increasing index column to it.

        Returns:
            DataFrame: The Spark DataFrame with an added index column.
        """
        """Load the Spark DataFrame from the input table and add an index column."""
        self.source_sdf = (self.spark.table(self.config.input_table_name)
                           .withColumn(self.index_column, monotonically_increasing_id()))
        return self.source_sdf

    def select_and_limit_data(self, sdf: DataFrame) -> DataFrame:
        """
        Select required columns from the input DataFrame and optionally limit the number of rows.

        Args:
            sdf (DataFrame): The input Spark DataFrame.

        Returns:
            DataFrame: A DataFrame with the selected columns and limited number of rows if specified.
        """
        result = sdf.select(self.index_column, self.config.input_column_name)
        if self.config.input_num_rows:
            result = result.limit(self.config.input_num_rows)
        self.input_sdf = result
        return self.input_sdf

    def convert_to_list(self, sdf: DataFrame) -> List[Tuple[int, str]]:
        """
        Convert a Spark DataFrame to a list of tuples.

        Args:
            sdf (DataFrame): The Spark DataFrame to convert.

        Returns:
            List[Tuple[int, str]]: A list of tuples where each tuple represents a row in the DataFrame.
        """
        pandas_df = sdf.toPandas()
        return [row for row in pandas_df.itertuples(index=False, name=None)]

    def process(self) -> List[Tuple[int, str]]:
        """
        Main method to process the data.

        This method performs the following steps:
        1. Loads a Spark DataFrame from the source.
        2. Selects and limits the data from the loaded DataFrame.
        3. Converts the selected data to a list of tuples containing an index and a string.

        Returns:
            List[Tuple[int, str]]: A list of tuples where each tuple contains an index and a string.
        """
        self.source_sdf = self.load_spark_dataframe()
        self.input_sdf = self.select_and_limit_data(self.source_sdf)
        self.texts_with_index = self.convert_to_list(self.input_sdf)
        return self.texts_with_index

    def get_source_sdf(self) -> Optional[DataFrame]:
        """
        Retrieve the source Spark DataFrame.

        Returns:
            Optional[DataFrame]: The source Spark DataFrame if available, otherwise None.
        """

        return self.source_sdf

    def get_input_sdf(self) -> Optional[DataFrame]:
        """
        Retrieve the input Spark DataFrame.

        Returns:
            Optional[DataFrame]: The input Spark DataFrame if available, otherwise None.
        """

        return self.input_sdf

    def get_texts_with_index(self) -> Optional[List[Tuple[int, str]]]:
        """
        Retrieve the processed list of texts along with their indices.

        Returns:
            Optional[List[Tuple[int, str]]]: A list of tuples where each tuple contains an index (int) and a text (str).
        """

        return self.texts_with_index