import os
import traceback
import pandas as pd
from tqdm import tqdm
from .commands import ExternalSaveCommands
from ..base.commands import DatasetCommands
from dataclasses import dataclass

@dataclass
class HDF5Params:
    ext_file: str = "data.h5"
    cache_size: int = 100
    root: str = ""

class HDF5Commands(ExternalSaveCommands):
    def __init__(self, arguments,split=None):
        super().__init__(arguments,split)
        self.hdf5_file = self.params.ext_file
        os.makedirs(os.path.dirname(self.hdf5_file), exist_ok=True)

    def save_info_to_hdf5(self, index):
        df = pd.DataFrame([self.info])
        df.to_hdf(self.hdf5_file, key=f"info_{index}", mode='a', format='table')
        
    def SaveRowToHDF5(self, index):
        self.ConvertforExternalSaving(self.data.loc[index], index)
        self.save_info_to_hdf5(index)

    def SaveAllToHDF5(self):
        for index in tqdm(self.data.index.tolist(), total=len(self.data.index.tolist()), desc="Saving to HDF5"):
            self.SaveRowToHDF5(index)

    def LoadHDF5DataFrame(self, file_path):
        try:
            self.data = pd.read_hdf(file_path)
        except Exception as e:
            print(f"Failed to load DataFrame from HDF5 file {file_path}")
            print(traceback.print_exc())

    def GetAllIndices(self):
        try:
            indices = self.data["Indices"].unique().tolist()
            return indices
        except Exception as e:
            print(f"Failed to retrieve indices from DataFrame")
            print(traceback.print_exc())
            return []

    def loadhdf5asinfodict(self, file_path, index):
        try:
            df = pd.read_hdf(file_path, key=f"info_{index}")
            self.info = df.to_dict(orient="list")
        except Exception as e:
            print(f"Failed to load info from HDF5 file {file_path} for index {index}")
            print(traceback.print_exc())

    def GetInfo(self, index):
        self.loadhdf5asinfodict(self.hdf5_file, index)
            
    def SampleHDF5(self, index):
        self.GetInfo(index)
        self.creategraphsample()
        return self.sample

    def FingerprintModelSamplerHDF5(self, index):
        self.GetInfo(index)
        self.creategraphsample()
        return self.sample


class GraphHDF5Dataset(HDF5Commands):
    def __init__(self, arguments,split=None):
        super().__init__(arguments,split)
        self.GetAllIndices()
    
    def __getitem__(self, index):
        if index in self.cache:
            samples = self.cache[index]
        else:
            try:
                samples = self.SampleHDF5(index)            
                if len(self.cache) < self.cache_size:
                    self.cache[index] = samples
            except Exception as e:
                print(self.root + '--'+ str(index) + ' failed')
                print(traceback.print_exc())
                return None
            
class FingerprintHDF5Dataset(HDF5Commands):
    def __init__(self, arguments,split=None):
        super().__init__(arguments,split)
        self.GetAllIndices()
    
    def __getitem__(self, index):
        if index in self.cache:
            samples = self.cache[index]
        else:
            try:
                samples = self.FingerprintModelSamplerHDF5(index)            
                if len(self.cache) < self.cache_size:
                    self.cache[index] = samples
            except Exception as e:
                print(self.root + '--'+ str(index) + ' failed')
                print(traceback.print_exc())
                return None

class HDF5Saver(HDF5Commands):
    def __init__(self, arguments,split=None):
        super().__init__(arguments,split)
        self.SaveAllToHDF5()
