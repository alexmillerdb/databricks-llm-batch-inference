import uuid
from typing import List, Tuple, Optional
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import monotonically_increasing_id
import pandas as pd

import os
import sys

current_directory = os.getcwd()
root_directory = os.path.normpath(os.path.join(current_directory, '..', '..'))
sys.path.append(root_directory)

from src.config.data_processor_config import DataProcessorConfig

class DataProcessor:
    def __init__(self, spark: SparkSession, config: DataProcessorConfig):
        self.spark = spark
        self.config = config
        self.index_column = f"index_{uuid.uuid4().hex[:4]}"
        self.source_sdf: Optional[DataFrame] = None
        self.input_sdf: Optional[DataFrame] = None
        self.texts_with_index: Optional[List[Tuple[int, str]]] = None

    def load_spark_dataframe(self) -> DataFrame:
        """Load the Spark DataFrame from the input table and add an index column."""
        self.source_sdf = (self.spark.table(self.config.input_table_name)
                           .withColumn(self.index_column, monotonically_increasing_id()))
        return self.source_sdf

    def select_and_limit_data(self, sdf: DataFrame) -> DataFrame:
        """Select required columns and optionally limit the number of rows."""
        result = sdf.select(self.index_column, self.config.input_column_name)
        if self.config.input_num_rows:
            result = result.limit(self.config.input_num_rows)
        self.input_sdf = result
        return self.input_sdf

    def convert_to_list(self, sdf: DataFrame) -> List[Tuple[int, str]]:
        """Convert Spark DataFrame to a list of tuples."""
        pandas_df = sdf.toPandas()
        return [row for row in pandas_df.itertuples(index=False, name=None)]

    def process(self) -> List[Tuple[int, str]]:
        """Main method to process the data."""
        self.source_sdf = self.load_spark_dataframe()
        self.input_sdf = self.select_and_limit_data(self.source_sdf)
        self.texts_with_index = self.convert_to_list(self.input_sdf)
        return self.texts_with_index

    def get_source_sdf(self) -> Optional[DataFrame]:
        """Get the source Spark DataFrame."""
        return self.source_sdf

    def get_input_sdf(self) -> Optional[DataFrame]:
        """Get the input Spark DataFrame."""
        return self.input_sdf

    def get_texts_with_index(self) -> Optional[List[Tuple[int, str]]]:
        """Get the processed list of texts with index."""
        return self.texts_with_index