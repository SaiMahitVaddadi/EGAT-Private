import traceback
from tqdm import tqdm
from .commands import ExternalSaveCommands
import pandas as pd


class PandasCommands(ExternalSaveCommands):
    def __init__(self, arguments,split=None):
        super().__init__(arguments,split)

    def setinfointoparquet(self, index):
        file_name = f"info_{index}.parquet"
        df = pd.DataFrame([self.info])
        df.to_parquet(file_name, index=False)

    def setinfolistintoparquet(self, index):
        file_name = f"info_{index}.parquet"
        df = pd.DataFrame({"all_result": self.infolist})
        df.to_parquet(file_name, index=False)

    def setinfointoparquetasseries(self, index):
        for i, info in enumerate(self.infolist):
            file_name = f"info_{index}_{i}.parquet"
            df = pd.DataFrame([info])
            df.to_parquet(file_name, index=False)

    def SaveRowToPandas(self, index):
        try:
            infolistusage = self.OrganizeData(index)

            if infolistusage:
                if self.params.saveconfsseparately:
                    self.setinfointoparquetasseries(index)
                else:
                    self.setinfolistintoparquet(index)
            else:
                self.setinfointoparquet(index)
        except Exception as e:
            print(f"Failed to save row {index} to Pandas")
            print(traceback.print_exc())

    def SaveInfoToPandas(self):
        for index in tqdm(self.data["index"].tolist(), total=len(self.data["index"]), desc="Saving to Pandas"):
            self.SaveRowToPandas(index)

    def LoadPandasDataFrame(self, file_path):
        try:
            self.data = pd.read_parquet(file_path)
        except Exception as e:
            print(f"Failed to load Pandas DataFrame from {file_path}")
            print(traceback.print_exc())

    def GetAllIndices(self):
        try:
            indices = self.data["Indices"].unique().tolist()
            return indices
        except Exception as e:
            print(f"Failed to retrieve indices from DataFrame")
            print(traceback.print_exc())
            return []

    def loadparquetasinfodict(self, file_path):
        try:
            df = pd.read_parquet(file_path)
            self.info = df.to_dict(orient="list")
        except Exception as e:
            print(f"Failed to load info from Pandas file {file_path}")
            print(traceback.print_exc())

    def loadparquetasinfolist(self, file_path):
        try:
            df = pd.read_parquet(file_path)
            self.infolist = df["all_result"].tolist()
        except Exception as e:
            print(f"Failed to load info list from Pandas file {file_path}")
            print(traceback.print_exc())

    def GetInfo(self, index):
        infolistusage = self.__useinfolist()
        file_path = f"info_{index}.parquet"
        if infolistusage:
            if self.params.saveconfsseparately:
                self.loadparquetasinfodict(file_path)
            else:
                self.loadparquetasinfolist(file_path)
        else:
            self.loadparquetasinfodict(file_path)

    def SamplePandas(self, index):
        infolistusage = self.__useinfolist()
        self.GetInfo(index)
        self.samples = [] 
        self.samples += self._sampleindices(infolistusage)
        self.samples += self._samplereactiontype(infolistusage)
        self.samples = self.AddGraphsToSample(self.samples)
        self.samples += self._sampletargets(infolistusage)
        self.samples += self._sampleadditionals(infolistusage)
        self.samples += self.Addons()
        return self.samples

    def FingerprintModelSamplerPandas(self, index):
        infolistusage = self.__useinfolist()
        self.GetInfo(index)
        self.samples = [] 
        self.samples += self._sampleindices(infolistusage)
        self.samples += self._samplereactiontype(infolistusage)
        self.samples += self._sampletargets(infolistusage)
        self.samples += self._sampleadditionals(infolistusage)
        self.samples += self.Addons()
        return self.samples

        
class GraphPandasDataset(PandasCommands):
    def __init__(self, arguments,split=None):
        super().__init__(arguments,split)
        self.LoadPandasDataFrame()
        self.GetAllIndices()
    
    def __getitem__(self, index):
        if index in self.cache:
            samples = self.cache[index]
        else:
            try:
                samples = self.SamplePandas(index)            
                if len(self.cache) < self.cache_size:
                    self.cache[index] = samples
            except Exception as e:
                print(self.root + '--'+ str(index) + ' failed')
                print(traceback.print_exc())
                return None
            
class FingerprintPandasDataset(PandasCommands):
    def __init__(self, arguments,split=None):
        super().__init__(arguments,split)
        self.LoadPandasDataFrame()
        self.GetAllIndices()
    
    def __getitem__(self, index):
        if index in self.cache:
            samples = self.cache[index]
        else:
            try:
                samples = self.FingerprintModelSamplerPandas(index)            
                if len(self.cache) < self.cache_size:
                    self.cache[index] = samples
            except Exception as e:
                print(self.root + '--'+ str(index) + ' failed')
                print(traceback.print_exc())
                return None

class PandasSaver(PandasCommands):
    def __init__(self, arguments,split=None):
        super().__init__(arguments,split)
        self.SaveInfoToPandas()
