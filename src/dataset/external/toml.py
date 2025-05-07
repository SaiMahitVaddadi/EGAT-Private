import os
import toml
import traceback
from tqdm import tqdm
from commands import ExternalSaveCommands
from ..base.commands import DatasetCommands
import pandas as pd

class TOMLCommands(ExternalSaveCommands):
    def __init__(self, arguments,split=None):
        super().__init__(arguments,split)
        self.toml_folder = self.params.toml_folder
        os.makedirs(self.toml_folder, exist_ok=True)

    def save_info_to_toml(self, index):
        file_path = os.path.join(self.toml_folder, f"info_{index}.toml")
        with open(file_path, 'w') as toml_file:
            toml.dump(self.info, toml_file)

    def save_info_list_to_toml(self, index):
        file_path = os.path.join(self.toml_folder, f"info_{index}.toml")
        with open(file_path, 'w') as toml_file:
            toml.dump({"all_result": self.infolist}, toml_file)

    def save_info_as_series_to_toml(self, index):
        for i, info in enumerate(self.infolist):
            file_path = os.path.join(self.toml_folder, f"info_{index}_{i}.toml")
            with open(file_path, 'w') as toml_file:
                toml.dump(info, toml_file)

    def SaveRowToTOML(self, index):
        try:
            infolistusage = self.OrganizeData(index)

            if infolistusage:
                if self.params.saveconfsseparately:
                    self.save_info_as_series_to_toml(index)
                else:
                    self.save_info_list_to_toml(index)
            else:
                self.save_info_to_toml(index)
        except Exception as e:
            self.__ExternalException(index)

    def SaveInfoToTOML(self):
        for index in tqdm(self.data.index.tolist(), total=len(self.data.index.tolist()), desc="Saving to TOML"):
            self.SaveRowToTOML(index)

    def load_toml_as_info_dict(self, index, i=None):
        try:
            if i is None:
                file_path = os.path.join(self.toml_folder, f"info_{index}.toml")
            else:
                file_path = os.path.join(self.toml_folder, f"info_{index}_{i}.toml")
            with open(file_path, 'r') as toml_file:
                self.info = toml.load(toml_file)
        except Exception as e:
            print(f"Failed to load info from {file_path}")
            print(traceback.print_exc())

    def load_toml_as_info_list(self, index):
        try:
            file_path = os.path.join(self.toml_folder, f"info_{index}.toml")
            with open(file_path, 'r') as toml_file:
                data = toml.load(toml_file)
                self.infolist = data.get("all_result", [])
        except Exception as e:
            print(f"Failed to load info list from {file_path}")
            print(traceback.print_exc())

    def GetInfo(self, index):
        infolistusage = self.__useinfolist()
        if infolistusage:
            if self.params.saveconfsseparately:
                self.load_toml_as_info_dict(index)
            else:
                self.load_toml_as_info_list(index)
        else:
            self.load_toml_as_info_dict(index)

    def SampleTOML(self, index):
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
    
    def FingerprintModelSamplerTOML(self, index):
        infolistusage = self.__useinfolist()
        self.GetInfo(index)
        self.samples = [] 
        self.samples += self._sampleindices(infolistusage)
        self.samples += self._samplereactiontype(infolistusage)
        samples += self._sampletargets(infolistusage)
        samples += self._sampleadditionals(infolistusage)
        samples += self.Addons()
        return samples


class GraphTOMLDataset(TOMLCommands):
    def __init__(self, arguments,split=None):
        super().__init__(arguments,split)
        self.GetAllIndices()
    
    def __getitem__(self, index):
        if index in self.cache:
            samples = self.cache[index]
        else:
            try:
                samples = self.SampleTOML(index)            
                if len(self.cache) < self.cache_size:
                    self.cache[index] = samples
            except Exception as e:
                print(self.root + '--'+ str(index) + ' failed')
                print(traceback.print_exc())
                return None
            
class FingerprintTOMLDataset(TOMLCommands):
    def __init__(self, arguments,split=None):
        super().__init__(arguments,split)
        self.GetAllIndices()
    
    def __getitem__(self, index):
        if index in self.cache:
            samples = self.cache[index]
        else:
            try:
                samples = self.FingerprintModelSamplerTOML(index)            
                if len(self.cache) < self.cache_size:
                    self.cache[index] = samples
            except Exception as e:
                print(self.root + '--'+ str(index) + ' failed')
                print(traceback.print_exc())
                return None

class TOMLSaver(TOMLCommands):
    def __init__(self, arguments,split=None):
        super().__init__(arguments,split)
        self.SaveInfoToTOML()
