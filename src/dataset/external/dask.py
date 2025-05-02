import traceback
from tqdm import tqdm
from commands import ExternalSaveCommands
import dask.dataframe as dd


class DaskCommands(ExternalSaveCommands):
    def __init__(self, arguments):
        super().__init__(arguments)

    def setinfointoparquet(self, index):
        file_name = f"info_{index}.parquet"
        df = dd.from_pandas(pd.DataFrame([self.info]), npartitions=1)
        df.to_parquet(file_name, write_index=False)

    def setinfolistintoparquet(self, index):
        file_name = f"info_{index}.parquet"
        df = dd.from_pandas(pd.DataFrame({"all_result": self.infolist}), npartitions=1)
        df.to_parquet(file_name, write_index=False)

    def setinfointoparquetasseries(self, index):
        for i, info in enumerate(self.infolist):
            file_name = f"info_{index}_{i}.parquet"
            df = dd.from_pandas(pd.DataFrame([info]), npartitions=1)
            df.to_parquet(file_name, write_index=False)

    def SaveRowToDask(self, index):
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
            print(f"Failed to save row {index} to Dask")
            print(traceback.print_exc())

    def SaveInfoToDask(self):
        for index in tqdm(self.data["index"].compute().tolist(), total=len(self.data["index"]), desc="Saving to Dask"):
            self.SaveRowToDask(index)

    def LoadDaskDataFrame(self, file_path):
        try:
            self.data = dd.read_parquet(file_path)
        except Exception as e:
            print(f"Failed to load Dask DataFrame from {file_path}")
            print(traceback.print_exc())

    def GetAllIndices(self):
        try:
            indices = self.data["Indices"].unique().compute().tolist()
            return indices
        except Exception as e:
            print(f"Failed to retrieve indices from DataFrame")
            print(traceback.print_exc())
            return []

    def loadparquetasinfodict(self, file_path):
        try:
            df = dd.read_parquet(file_path)
            self.info = df.compute().to_dict(orient="list")
        except Exception as e:
            print(f"Failed to load info from Dask file {file_path}")
            print(traceback.print_exc())

    def loadparquetasinfolist(self, file_path):
        try:
            df = dd.read_parquet(file_path)
            self.infolist = df["all_result"].compute().tolist()
        except Exception as e:
            print(f"Failed to load info list from Dask file {file_path}")
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

    def SampleDask(self, index):
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

    def FingerprintModelSamplerDask(self, index):
        infolistusage = self.__useinfolist()
        self.GetInfo(index)
        self.samples = [] 
        self.samples += self._sampleindices(infolistusage)
        self.samples += self._samplereactiontype(infolistusage)
        self.samples += self._sampletargets(infolistusage)
        self.samples += self._sampleadditionals(infolistusage)
        self.samples += self.Addons()
        return self.samples

        
class GraphDaskDataset(DaskCommands):
    def __init__(self, arguments):
        super().__init__(arguments)
        self.LoadDaskDataFrame()
        self.GetAllIndices()
    
    def __getitem__(self, index):
        if index in self.cache:
            samples = self.cache[index]
        else:
            try:
                samples = self.SampleDask(index)            
                if len(self.cache) < self.cache_size:
                    self.cache[index] = samples
            except Exception as e:
                print(self.root + '--'+ str(index) + ' failed')
                print(traceback.print_exc())
                return None
            
class FingerprintDaskDataset(DaskCommands):
    def __init__(self, arguments):
        super().__init__(arguments)
        self.LoadDaskDataFrame()
        self.GetAllIndices()
    
    def __getitem__(self, index):
        if index in self.cache:
            samples = self.cache[index]
        else:
            try:
                samples = self.FingerprintModelSamplerDask(index)            
                if len(self.cache) < self.cache_size:
                    self.cache[index] = samples
            except Exception as e:
                print(self.root + '--'+ str(index) + ' failed')
                print(traceback.print_exc())
                return None

class DaskSaver(DaskCommands):
    def __init__(self, arguments):
        super().__init__(arguments)
        self.SaveInfoToDask()
