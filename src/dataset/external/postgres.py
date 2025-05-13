import traceback
from tqdm import tqdm
from .commands import ExternalSaveCommands
from pyspark.sql import SparkSession
import os
from dataclasses import dataclass, field
from typing import Optional, Dict

@dataclass
class PostgresQLParams:
    app_name: str = "PostgresQLApplication"
    master: str = "local[*]"
    spark_jars: str = "/path/to/postgresql.jar"
    url: str = "jdbc:postgresql://<host>:<port>/<database>"
    user: str = "<username>"
    password: str = "<password>"
    additional_options: Dict[str, str] = field(default_factory=dict)
class PostgresQLCommands(ExternalSaveCommands):
    def __init__(self, arguments, split=None):
        super().__init__(arguments, split)

    def __init__(self, arguments, split=None):
        super().__init__(arguments, split)
        self.spark_session = self.create_spark_session()

    def create_spark_session(self):
        """
        Creates and returns a Spark session configured for PostgreSQL.

        Returns:
        SparkSession: A configured Spark session.
        """
        app_name = self.params.get("app_name", "PostgresQLApplication")
        master = self.params.get("master", "local[*]")
        jars_path = self.params.get("spark_jars", "/path/to/postgresql.jar")
        return SparkSession.builder \
            .appName(app_name) \
            .master(master) \
            .config("spark.jars", jars_path) \
            .getOrCreate()

    def save_info_to_postgres(self, index):
        table_name = f"info_{index}"
        df = self.spark_session.createDataFrame([self.info])
        df.write \
            .format("jdbc") \
            .option("url", self.params.get("url", "jdbc:postgresql://<host>:<port>/<database>")) \
            .option("dbtable", table_name) \
            .option("user", self.params.get("user", "<username>")) \
            .option("password", self.params.get("password", "<password>")) \
            .mode("overwrite") \
            .save()

    def SaveRowToPostgresQL(self, index):
        self.ConvertforExternalSaving(self.data.loc[index], index)
        self.save_info_to_postgres(index)

    def SaveAllToPostgresQL(self):
        for index in tqdm(self.data.index.tolist(), total=len(self.data.index.tolist()), desc="Saving to PostgresQL"):
            self.SaveRowToPostgresQL(index)

    def LoadPostgresQLDataFrame(self):
        try:
            self.data = self.spark_session.sql("SELECT * FROM data").toPandas()
        except Exception as e:
            print(f"Failed to load PostgresQL DataFrame")
            print(traceback.print_exc())

    def GetAllIndices(self):
        try:
            indices = self.data["Indices"].unique().tolist()
            return indices
        except Exception as e:
            print(f"Failed to retrieve indices from DataFrame")
            print(traceback.print_exc())
            return []

    def load_postgres_as_info_dict(self, index):
        try:
            table_name = f"info_{index}"
            query = f"(SELECT * FROM {table_name}) AS subquery"
            df = self.spark_session.read \
                .format("jdbc") \
                .option("url", self.params.get("url", "jdbc:postgresql://<host>:<port>/<database>")) \
                .option("dbtable", query) \
                .option("user", self.params.get("user", "<username>")) \
                .option("password", self.params.get("password", "<password>")) \
                .load()
            self.info = df.toPandas().to_dict(orient="list")
        except Exception as e:
            print(f"Failed to load info from PostgresQL table {table_name}")
            print(traceback.print_exc())

    def GetInfo(self, index):
        self.load_postgres_as_info_dict(index)

    def SamplePostgresQL(self, index):
        self.GetInfo(index)
        self.creategraphsample()
        return self.sample

    def FingerprintModelSamplerPostgresQL(self, index):
        self.GetInfo(index)
        self.creategraphsample()
        return self.sample


class GraphPostgresQLDataset(PostgresQLCommands):
    def __init__(self, arguments, split=None):
        super().__init__(arguments, split)
        self.LoadPostgresQLDataFrame()
        self.GetAllIndices()

    def __getitem__(self, index):
        if index in self.cache:
            samples = self.cache[index]
        else:
            try:
                spark_session = self.create_spark_session()
                samples = self.SamplePostgresQL(index, spark_session)
                if len(self.cache) < self.cache_size:
                    self.cache[index] = samples
            except Exception as e:
                print(self.root + '--' + str(index) + ' failed')
                print(traceback.print_exc())
                return None


class FingerprintPostgresQLDataset(PostgresQLCommands):
    def __init__(self, arguments, split=None):
        super().__init__(arguments, split)
        self.LoadPostgresQLDataFrame()
        self.GetAllIndices()

    def __getitem__(self, index):
        if index in self.cache:
            samples = self.cache[index]
        else:
            try:
                spark_session = self.create_spark_session()
                samples = self.FingerprintModelSamplerPostgresQL(index, spark_session)
                if len(self.cache) < self.cache_size:
                    self.cache[index] = samples
            except Exception as e:
                print(self.root + '--' + str(index) + ' failed')
                print(traceback.print_exc())
                return None


class PostgresQLSaver(PostgresQLCommands):
    def __init__(self, arguments, split=None):
        super().__init__(arguments, split)
        self.SaveAllToPostgresQL()
