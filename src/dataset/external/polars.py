import traceback
from tqdm import tqdm
from .commands import ExternalSaveCommands
import polars as pl


class ParquetCommands(ExternalSaveCommands):
    def __init__(self, arguments,split=None):
        super().__init__(arguments,split)

    def setinfointoparquet(self, index):
        file_name = f"info_{index}.parquet"
        df = pl.DataFrame([self.info])
        df.write_parquet(file_name)

    def setinfolistintoparquet(self, index):
        file_name = f"info_{index}.parquet"
        df = pl.DataFrame({"all_result": self.infolist})
        df.write_parquet(file_name)

    def setinfointoparquetasseries(self, index):
        for i, info in enumerate(self.infolist):
            file_name = f"info_{index}_{i}.parquet"
            df = pl.DataFrame([info])
            df.write_parquet(file_name)

    def SaveRowToParquet(self, index):
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
            print(f"Failed to save row {index} to Parquet")
            print(traceback.print_exc())

    def SaveInfoToParquet(self):
        for index in tqdm(self.data["index"].to_list(), total=len(self.data["index"]), desc="Saving to Parquet"):
            self.SaveRowToParquet(index)

    def LoadParquetDataFrame(self, file_path):
        try:
            self.data = pl.read_parquet(file_path)
        except Exception as e:
            print(f"Failed to load Parquet DataFrame from {file_path}")
            print(traceback.print_exc())

    def GetAllIndices(self):
        try:
            indices = self.data["Indices"].unique().to_list()
            return indices
        except Exception as e:
            print(f"Failed to retrieve indices from DataFrame")
            print(traceback.print_exc())
            return []

    def loadparquetasinfodict(self, file_path):
        try:
            df = pl.read_parquet(file_path)
            self.info = {col: df[col].to_list() for col in df.columns}
        except Exception as e:
            print(f"Failed to load info from Parquet file {file_path}")
            print(traceback.print_exc())

    def loadparquetasinfolist(self, file_path):
        try:
            df = pl.read_parquet(file_path)
            self.infolist = df["all_result"].to_list()
        except Exception as e:
            print(f"Failed to load info list from Parquet file {file_path}")
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

    def SampleParquet(self, index):
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

    def FingerprintModelSamplerParquet(self, index):
        infolistusage = self.__useinfolist()
        self.GetInfo(index)
        self.samples = [] 
        self.samples += self._sampleindices(infolistusage)
        self.samples += self._samplereactiontype(infolistusage)
        self.samples += self._sampletargets(infolistusage)
        self.samples += self._sampleadditionals(infolistusage)
        self.samples += self.Addons()
        return self.samples

        
class GraphParquetDataset(ParquetCommands):
    def __init__(self, arguments,split=None):
        super().__init__(arguments,split)
        self.LoadParquetDataFrame()
        self.GetAllIndices()
    
    def __getitem__(self, index):
        if index in self.cache:
            samples = self.cache[index]
        else:
            try:
                samples = self.SampleParquet(index)            
                if len(self.cache) < self.cache_size:
                    self.cache[index] = samples
            except Exception as e:
                print(self.root + '--'+ str(index) + ' failed')
                print(traceback.print_exc())
                return None
            
class FingerprintParquetDataset(ParquetCommands):
    def __init__(self, arguments,split=None):
        super().__init__(arguments,split)
        self.LoadParquetDataFrame()
        self.GetAllIndices()
    
    def __getitem__(self, index):
        if index in self.cache:
            samples = self.cache[index]
        else:
            try:
                samples = self.FingerprintModelSamplerParquet(index)            
                if len(self.cache) < self.cache_size:
                    self.cache[index] = samples
            except Exception as e:
                print(self.root + '--'+ str(index) + ' failed')
                print(traceback.print_exc())
                return None

class ParquetSaver(ParquetCommands):
    def __init__(self, arguments,split=None):
        super().__init__(arguments,split)
        self.SaveInfoToParquet()




 

