import os
import json
import traceback
from tqdm import tqdm
from commands import ExternalSaveCommands
from ..base.commands import DatasetCommands
import pandas as pd

class JSONCommands(ExternalSaveCommands):
    def __init__(self, arguments):
        super().__init__(arguments)
        self.json_folder = self.params.json_folder
        os.makedirs(self.json_folder, exist_ok=True)

    def save_info_to_json(self, index):
        file_path = os.path.join(self.json_folder, f"info_{index}.json")
        with open(file_path, 'w') as json_file:
            json.dump(self.info, json_file, indent=4)

    def save_info_list_to_json(self, index):
        file_path = os.path.join(self.json_folder, f"info_{index}.json")
        with open(file_path, 'w') as json_file:
            json.dump({"all_result": self.infolist}, json_file, indent=4)

    def save_info_as_series_to_json(self, index):
        for i, info in enumerate(self.infolist):
            file_path = os.path.join(self.json_folder, f"info_{index}_{i}.json")
            with open(file_path, 'w') as json_file:
                json.dump(info, json_file, indent=4)

    def SaveRowToJSON(self, index):
        try:
            infolistusage = self.OrganizeData(index)

            if infolistusage:
                if self.params.saveconfsseparately:
                    self.save_info_as_series_to_json(index)
                else:
                    self.save_info_list_to_json(index)
            else:
                self.save_info_to_json(index)
        except Exception as e:
            self.__ExternalException(index)

    def SaveInfoToJSON(self):
        for index in tqdm(self.data.index.tolist(), total=len(self.data.index.tolist()), desc="Saving to JSON"):
            self.SaveRowToJSON(index)

    def load_json_as_info_dict(self, index, i=None):
        try:
            if i is None:
                file_path = os.path.join(self.json_folder, f"info_{index}.json")
            else:
                file_path = os.path.join(self.json_folder, f"info_{index}_{i}.json")
            with open(file_path, 'r') as json_file:
                self.info = json.load(json_file)
        except Exception as e:
            print(f"Failed to load info from {file_path}")
            print(traceback.print_exc())

    def load_json_as_info_list(self, index):
        try:
            file_path = os.path.join(self.json_folder, f"info_{index}.json")
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
                self.infolist = data.get("all_result", [])
        except Exception as e:
            print(f"Failed to load info list from {file_path}")
            print(traceback.print_exc())

    def GetInfo(self, index):
        infolistusage = self.__useinfolist()
        if infolistusage:
            if self.params.saveconfsseparately:
                self.load_json_as_info_dict(index)
            else:
                self.load_json_as_info_list(index)
        else:
            self.load_json_as_info_dict(index)

    def SampleJSON(self, index):
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
    
    def FingerprintModelSamplerJSON(self, index):
        infolistusage = self.__useinfolist()
        self.GetInfo(index)
        self.samples = [] 
        self.samples += self._sampleindices(infolistusage)
        self.samples += self._samplereactiontype(infolistusage)
        samples += self._sampletargets(infolistusage)
        samples += self._sampleadditionals(infolistusage)
        samples += self.Addons()
        return samples


class GraphJSONDataset(JSONCommands):
    def __init__(self, arguments):
        super().__init__(arguments)
        self.GetAllIndices()
    
    def __getitem__(self, index):
        if index in self.cache:
            samples = self.cache[index]
        else:
            try:
                samples = self.SampleJSON(index)            
                if len(self.cache) < self.cache_size:
                    self.cache[index] = samples
            except Exception as e:
                print(self.root + '--'+ str(index) + ' failed')
                print(traceback.print_exc())
                return None
            
class FingerprintJSONDataset(JSONCommands):
    def __init__(self, arguments):
        super().__init__(arguments)
        self.GetAllIndices()
    
    def __getitem__(self, index):
        if index in self.cache:
            samples = self.cache[index]
        else:
            try:
                samples = self.FingerprintModelSamplerJSON(index)            
                if len(self.cache) < self.cache_size:
                    self.cache[index] = samples
            except Exception as e:
                print(self.root + '--'+ str(index) + ' failed')
                print(traceback.print_exc())
                return None

class JSONSaver(JSONCommands):
    def __init__(self, arguments):
        super().__init__(arguments)
        self.SaveInfoToJSON()
