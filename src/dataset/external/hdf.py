import os
import traceback
import pandas as pd
from tqdm import tqdm
from commands import ExternalSaveCommands
from ..base.commands import DatasetCommands

class HDF5Commands(ExternalSaveCommands):
    def __init__(self, arguments):
        super().__init__(arguments)
        self.hdf5_file = self.params.hdf5_file
        os.makedirs(os.path.dirname(self.hdf5_file), exist_ok=True)

    def save_info_to_hdf5(self, index):
        try:
            df = pd.DataFrame([self.info])
            df.to_hdf(self.hdf5_file, key=f"info_{index}", mode='a', format='table')
        except Exception as e:
            print(f"Failed to save info for index {index} to HDF5")
            print(traceback.print_exc())

    def save_info_list_to_hdf5(self, index):
        try:
            df = pd.DataFrame(self.infolist)
            df.to_hdf(self.hdf5_file, key=f"info_{index}", mode='a', format='table')
        except Exception as e:
            print(f"Failed to save info list for index {index} to HDF5")
            print(traceback.print_exc())

    def SaveRowToHDF5(self, index):
        try:
            infolistusage = self.OrganizeData(index)

            if infolistusage:
                self.save_info_list_to_hdf5(index)
            else:
                self.save_info_to_hdf5(index)
        except Exception as e:
            self.__ExternalException(index)

    def SaveInfoToHDF5(self):
        for index in tqdm(self.data.index.tolist(), total=len(self.data.index.tolist()), desc="Saving to HDF5"):
            self.SaveRowToHDF5(index)

    def load_hdf5_as_info_dict(self, index):
        try:
            self.info = pd.read_hdf(self.hdf5_file, key=f"info_{index}").to_dict(orient='records')[0]
        except Exception as e:
            print(f"Failed to load info for index {index} from HDF5")
            print(traceback.print_exc())

    def load_hdf5_as_info_list(self, index):
        try:
            self.infolist = pd.read_hdf(self.hdf5_file, key=f"info_{index}").to_dict(orient='records')
        except Exception as e:
            print(f"Failed to load info list for index {index} from HDF5")
            print(traceback.print_exc())

    def GetInfo(self, index):
        infolistusage = self.__useinfolist()
        if infolistusage:
            self.load_hdf5_as_info_list(index)
        else:
            self.load_hdf5_as_info_dict(index)

    def SampleHDF5(self, index):
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
    
    def FingerprintModelSamplerHDF5(self, index):
        infolistusage = self.__useinfolist()
        self.GetInfo(index)
        self.samples = [] 
        self.samples += self._sampleindices(infolistusage)
        self.samples += self._samplereactiontype(infolistusage)
        samples += self._sampletargets(infolistusage)
        samples += self._sampleadditionals(infolistusage)
        samples += self.Addons()
        return samples


class GraphHDF5Dataset(HDF5Commands):
    def __init__(self, arguments):
        super().__init__(arguments)
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
    def __init__(self, arguments):
        super().__init__(arguments)
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
    def __init__(self, arguments):
        super().__init__(arguments)
        self.SaveInfoToHDF5()
