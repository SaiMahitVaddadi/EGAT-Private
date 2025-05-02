import traceback
from tqdm import tqdm
from commands import ExternalSaveCommands
from pyspark.sql import SparkSession
class PostgresQLCommands(ExternalSaveCommands):
    def __init__(self, arguments):
        super().__init__(arguments)

    def create_spark_session(app_name="PostgresQLApplication", master="local[*]"):
        """
        Creates and returns a Spark session configured for PostgreSQL.

        Parameters:
        app_name (str): Name of the Spark application.
        master (str): The master URL for the cluster.

        Returns:
        SparkSession: A configured Spark session.
        """
        return SparkSession.builder \
            .appName(app_name) \
            .master(master) \
            .config("spark.jars", "/path/to/postgresql.jar") \
            .getOrCreate()

    def setinfointopostgres(self, index, spark_session):
        table_name = f"info_{index}"
        df = spark_session.createDataFrame([self.info])
        df.write \
            .format("jdbc") \
            .option("url", "jdbc:postgresql://<host>:<port>/<database>") \
            .option("dbtable", table_name) \
            .option("user", "<username>") \
            .option("password", "<password>") \
            .mode("overwrite") \
            .save()

    def setinfolistintopostgres(self, index, spark_session):
        table_name = f"info_{index}"
        df = spark_session.createDataFrame(self.infolist, schema=["all_result"])
        df.write \
            .format("jdbc") \
            .option("url", "jdbc:postgresql://<host>:<port>/<database>") \
            .option("dbtable", table_name) \
            .option("user", "<username>") \
            .option("password", "<password>") \
            .mode("overwrite") \
            .save()

    def setinfointopostgresasseries(self, index, spark_session):
        for i, info in enumerate(self.infolist):
            table_name = f"info_{index}_{i}"
            df = spark_session.createDataFrame([info])
            df.write \
                .format("jdbc") \
                .option("url", "jdbc:postgresql://<host>:<port>/<database>") \
                .option("dbtable", table_name) \
                .option("user", "<username>") \
                .option("password", "<password>") \
                .mode("overwrite") \
                .save()

    def SaveRowToPostgresQL(self, index, spark_session):
        try:
            infolistusage = self.OrganizeData(index)

            if infolistusage:
                if self.params.saveconfsseparately:
                    self.setinfointopostgresasseries(index, spark_session)
                else:
                    self.setinfolistintopostgres(index, spark_session)
            else:
                self.setinfointopostgres(index, spark_session)
        except Exception as e:
            self.__ExternalException(index)

    def SaveInfoToPostgresQL(self):
        spark_session = self.create_spark_session()
        for index in tqdm(self.data.index.tolist(), total=len(self.data.index.tolist()), desc="Saving to PostgresQL"):
            self.SaveRowToPostgresQL(index, spark_session)

    def LoadPostgresQLDataFrame(self):
        spark_session = self.create_spark_session()
        try:
            self.data = spark_session.sql("SELECT * FROM data").toPandas()
        except Exception as e:
            print(f"Failed to load PostgresQL DataFrame")
            print(traceback.print_exc())

    def GetAllIndices(self, spark_session):
        try:
            indices_df = spark_session.sql("SELECT DISTINCT Indices FROM data")
            indices = [row.Indices for row in indices_df.collect()]
            return indices
        except Exception as e:
            print(f"Failed to retrieve indices from PostgresQL")
            print(traceback.print_exc())
            return []

    def loadpostgresasinfodict(self, index, i=None, spark_session=None):
        try:
            if spark_session is None:
                spark_session = self.create_spark_session()
            if i is None:
                table_name = f"info_{index}"
            else:
                table_name = f"info_{index}_{i}"
            query = f"(SELECT * FROM {table_name}) AS subquery"
            df = spark_session.read \
                .format("jdbc") \
                .option("url", "jdbc:postgresql://<host>:<port>/<database>") \
                .option("dbtable", query) \
                .option("user", "<username>") \
                .option("password", "<password>") \
                .load()
            rows = df.collect()
            self.info = {col: row[col] for row in rows for col in df.columns}
        except Exception as e:
            print(f"Failed to load info from PostgresQL table {table_name}")
            print(traceback.print_exc())

    def loadpostgresasinfolist(self, index, spark_session=None):
        try:
            if spark_session is None:
                spark_session = self.create_spark_session()
            table_name = f"info_{index}"
            query = f"(SELECT * FROM {table_name}) AS subquery"
            df = spark_session.read \
                .format("jdbc") \
                .option("url", "jdbc:postgresql://<host>:<port>/<database>") \
                .option("dbtable", query) \
                .option("user", "<username>") \
                .option("password", "<password>") \
                .load()
            rows = df.collect()
            self.infolist = [row[df.columns[0]] for row in rows]
        except Exception as e:
            print(f"Failed to load info list from PostgresQL table {table_name}")
            print(traceback.print_exc())

    def GetInfo(self, index, spark_session=None):
        infolistusage = self.__useinfolist()
        if spark_session is None:
            spark_session = self.create_spark_session()
        if infolistusage:
            if self.params.saveconfsseparately:
                self.loadpostgresasinfodict(index, spark_session=spark_session)
            else:
                self.loadpostgresasinfolist(index, spark_session=spark_session)
        else:
            self.loadpostgresasinfodict(index, spark_session=spark_session)

    def SamplePostgresQL(self,index):
        infolistusage = self.__useinfolist()
        self.GetInfo(index)
        self.samples = [] 
        self.samples += self._sampleindices(infolistusage)
        self.samples += self._samplereactiontype(infolistusage)
        self.samples = self.AddGraphsToSample(self.samples)
        samples += self._sampletargets(infolistusage)
        samples += self._sampleadditionals(infolistusage)
        samples += self.Addons()
        return samples
    
    def FingerprintModelSamplerPostgresQL(self,index):
        infolistusage = self.__useinfolist()
        self.GetInfo(index)
        self.samples = [] 
        self.samples += self._sampleindices(infolistusage)
        self.samples += self._samplereactiontype(infolistusage)
        samples += self._sampletargets(infolistusage)
        samples += self._sampleadditionals(infolistusage)
        samples += self.Addons()
        return samples

        
class GraphPostgresQLDataset(PostgresQLCommands):
    def __init__(self, arguments):
        super().__init__(arguments)
        self.LoadPostgresQLDataFrame()
        self.GetAllIndices()
    
    def __getitem__(self, index):
        if index in self.cache:
            samples = self.cache[index]
        else:
            try:
                samples = self.SamplePostgresQL(index)            
                if len(self.cache) < self.cache_size:
                    self.cache[index] = samples
            except Exception as e:
                print(self.root + '--'+ str(index) + ' failed')
                print(traceback.print_exc())
                return None
            
class FingerprintPostgresQLDataset(PostgresQLCommands):
    def __init__(self, arguments):
        super().__init__(arguments)
        self.LoadPostgresQLDataFrame()
        self.GetAllIndices()
    
    def __getitem__(self, index):
        if index in self.cache:
            samples = self.cache[index]
        else:
            try:
                samples = self.FingerprintModelSamplerPostgresQL(index)            
                if len(self.cache) < self.cache_size:
                    self.cache[index] = samples
            except Exception as e:
                print(self.root + '--'+ str(index) + ' failed')
                print(traceback.print_exc())
                return None

class PostgresQLSaver(PostgresQLCommands):
    def __init__(self, arguments):
        super().__init__(arguments)
        self.SaveInfoToPostgresQL()




 

