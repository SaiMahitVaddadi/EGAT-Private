import traceback
from tqdm import tqdm
from commands import ExternalSaveCommands
from pyspark.sql import SparkSession
class SparkCommands(ExternalSaveCommands):
    def __init__(self, arguments):
        super().__init__(arguments)

    def create_spark_session(app_name="SparkApplication", master="local[*]"):
        """
        Creates and returns a Spark session.

        Parameters:
        app_name (str): Name of the Spark application.
        master (str): The master URL for the cluster.

        Returns:
        SparkSession: A configured Spark session.
        """
        return SparkSession.builder \
            .appName(app_name) \
            .master(master) \
            .enableHiveSupport() \
            .getOrCreate()

    def setinfointospark(self, index, spark_session):
        table_name = f"info_{index}"
        df = spark_session.createDataFrame([self.info])
        df.write.mode("overwrite").saveAsTable(table_name)

    def setinfolistintospark(self, index, spark_session):
        table_name = f"info_{index}"
        df = spark_session.createDataFrame(self.infolist, schema=["all_result"])
        df.write.mode("overwrite").saveAsTable(table_name)

    def setinfointosparkasseries(self, index, spark_session):
        for i, info in enumerate(self.infolist):
            table_name = f"info_{index}_{i}"
            df = spark_session.createDataFrame([info])
            df.write.mode("overwrite").saveAsTable(table_name)

    def SaveRowToSpark(self, index, spark_session):
        try:
            infolistusage = self.OrganizeData(index)

            if infolistusage:
                if self.params.saveconfsseparately:
                    self.setinfointosparkasseries(index, spark_session)
                else:
                    self.setinfolistintospark(index, spark_session)
            else:
                self.setinfointospark(index, spark_session)
        except Exception as e:
            self.__ExternalException(index)

    def SaveInfoToSpark(self):
        spark_session = self.create_spark_session()
        for index in tqdm(self.data.index.tolist(), total=len(self.data.index.tolist()), desc="Saving to Spark"):
            self.SaveRowToSpark(index, spark_session)

    def LoadSparkDataFrame(self):
        spark_session = self.create_spark_session()
        try:
            self.data = spark_session.sql("SELECT * FROM data").toPandas()
        except Exception as e:
            print(f"Failed to load Spark DataFrame")
            print(traceback.print_exc())

    def GetAllIndices(self, spark_session):
        try:
            indices_df = spark_session.sql("SELECT DISTINCT Indices FROM data")
            indices = [row.Indices for row in indices_df.collect()]
            return indices
        except Exception as e:
            print(f"Failed to retrieve indices from Spark")
            print(traceback.print_exc())
            return []

    def loadsparkasinfodict(self, index, i=None, spark_session=None):
        try:
            if spark_session is None:
                spark_session = self.create_spark_session()
            if i is None:
                table_name = f"info_{index}"
            else:
                table_name = f"info_{index}_{i}"
            query = f"SELECT * FROM {table_name}"
            df = spark_session.sql(query)
            rows = df.collect()
            self.info = {col: row[col] for row in rows for col in df.columns}
        except Exception as e:
            print(f"Failed to load info from Spark table {table_name}")
            print(traceback.print_exc())

    def loadsparkasinfolist(self, index, spark_session=None):
        try:
            if spark_session is None:
                spark_session = self.create_spark_session()
            table_name = f"info_{index}"
            query = f"SELECT * FROM {table_name}"
            df = spark_session.sql(query)
            rows = df.collect()
            self.infolist = [row[df.columns[0]] for row in rows]
        except Exception as e:
            print(f"Failed to load info list from Spark table {table_name}")
            print(traceback.print_exc())

    def GetInfo(self, index, spark_session=None):
        infolistusage = self.__useinfolist()
        if spark_session is None:
            spark_session = self.create_spark_session()
        if infolistusage:
            if self.params.saveconfsseparately:
                self.loadsparkasinfodict(index, spark_session=spark_session)
            else:
                self.loadsparkasinfolist(index, spark_session=spark_session)
        else:
            self.loadsparkasinfodict(index, spark_session=spark_session)

    def SampleSpark(self,index):
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
    
    def FingerprintModelSamplerSpark(self,index):
        infolistusage = self.__useinfolist()
        self.GetInfo(index)
        self.samples = [] 
        self.samples += self._sampleindices(infolistusage)
        self.samples += self._samplereactiontype(infolistusage)
        samples += self._sampletargets(infolistusage)
        samples += self._sampleadditionals(infolistusage)
        samples += self.Addons()
        return samples

        
class GraphSparkDataset(SparkCommands):
    def __init__(self, arguments):
        super().__init__(arguments)
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
                print(self.root + '--'+ str(index) + ' failed')
                print(traceback.print_exc())
                return None
            
class FingerprintSparkDataset(SparkCommands):
    def __init__(self, arguments):
        super().__init__(arguments)
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
                print(self.root + '--'+ str(index) + ' failed')
                print(traceback.print_exc())
                return None

class SparkSaver(SparkCommands):
    def __init__(self, arguments):
        super().__init__(arguments)
        self.SaveInfoToSpark()




 

