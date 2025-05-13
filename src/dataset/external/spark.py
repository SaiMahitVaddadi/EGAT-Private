import traceback
from tqdm import tqdm
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from .commands import ExternalSaveCommands
from dataclasses import dataclass

@dataclass
class SparkParams:
    app_name: str = "SparkApplication"
    master: str = "local[*]"
class SparkCommands(ExternalSaveCommands):
    def __init__(self, arguments, split=None):
        super().__init__(arguments, split)
        self.spark_session = self.create_spark_session()

    def create_spark_session(self):
        """
        Creates and returns a Spark session.
        """
        return SparkSession.builder \
            .appName(self.params.app_name) \
            .master(self.params.master) \
            .enableHiveSupport() \
            .getOrCreate()

    def save_info_to_spark(self, index):
        """
        Save self.info to a Spark table.
        """
        table_name = f"info_{index}"
        df = self.spark_session.createDataFrame([self.info])
        df.write.mode("overwrite").saveAsTable(table_name)

    def SaveRowToSpark(self, index):
        """
        Save a single row to Spark.
        """
        self.ConvertforExternalSaving(self.data.loc[index], index)
        self.save_info_to_spark(index)

    def SaveAllToSpark(self):
        """
        Save all rows to Spark.
        """
        for index in tqdm(self.data.index.tolist(), total=len(self.data.index.tolist()), desc="Saving to Spark"):
            self.SaveRowToSpark(index)

    def LoadSparkDataFrame(self):
        """
        Load the main DataFrame from Spark.
        """
        try:
            self.data = self.spark_session.sql("SELECT * FROM data").toPandas()
        except Exception as e:
            print(f"Failed to load DataFrame from Spark")
            print(traceback.print_exc())

    def GetAllIndices(self):
        """
        Retrieve all unique indices from the Spark DataFrame.
        """
        try:
            indices_df = self.spark_session.sql("SELECT DISTINCT Indices FROM data")
            indices = [row.Indices for row in indices_df.collect()]
            return indices
        except Exception as e:
            print(f"Failed to retrieve indices from Spark")
            print(traceback.print_exc())
            return []

    def load_spark_as_info_dict(self, index):
        """
        Load a specific table from Spark as a dictionary.
        """
        try:
            table_name = f"info_{index}"
            query = f"SELECT * FROM {table_name}"
            df = self.spark_session.sql(query)
            rows = df.collect()
            self.info = {col: row[col] for row in rows for col in df.columns}
        except Exception as e:
            print(f"Failed to load info from Spark table {table_name}")
            print(traceback.print_exc())

    def GetInfo(self, index):
        """
        Retrieve info for a specific index.
        """
        self.load_spark_as_info_dict(index)

    def SampleSpark(self, index):
        """
        Sample data from Spark for a specific index.
        """
        self.GetInfo(index)
        self.creategraphsample()
        return self.sample

    def FingerprintModelSamplerSpark(self, index):
        """
        Sample fingerprint model data from Spark for a specific index.
        """
        self.GetInfo(index)
        self.creategraphsample()
        return self.sample


class GraphSparkDataset(SparkCommands):
    def __init__(self, arguments, split=None):
        super().__init__(arguments, split)
        self.LoadSparkDataFrame()
        self.GetAllIndices()

    def __getitem__(self, index):
        if index in self.cache:
            samples = self.cache[index]
        else:
            try:
                samples = self.SampleSpark(index)
                if len(self.cache) < self.cache_size:
                    self.cache[index] = samples
            except Exception as e:
                print(self.root + '--' + str(index) + ' failed')
                print(traceback.print_exc())
                return None


class FingerprintSparkDataset(SparkCommands):
    def __init__(self, arguments, split=None):
        super().__init__(arguments, split)
        self.LoadSparkDataFrame()
        self.GetAllIndices()

    def __getitem__(self, index):
        if index in self.cache:
            samples = self.cache[index]
        else:
            try:
                samples = self.FingerprintModelSamplerSpark(index)
                if len(self.cache) < self.cache_size:
                    self.cache[index] = samples
            except Exception as e:
                print(self.root + '--' + str(index) + ' failed')
                print(traceback.print_exc())
                return None


class SparkSaver(SparkCommands):
    def __init__(self, arguments, split=None):
        super().__init__(arguments, split)
        self.SaveAllToSpark()
