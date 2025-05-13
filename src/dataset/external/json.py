import os
import json
import traceback
import pandas as pd
from tqdm import tqdm
from .commands import ExternalSaveCommands
from ..base.commands import DatasetCommands
from dataclasses import dataclass, field

@dataclass
class JSONParams:
    ext_file: str = field(default="json_data")
    cache_size: int = field(default=100)
    root: str = field(default=".")
    other_param: str = field(default="default_value")

class JSONCommands(ExternalSaveCommands):
    def __init__(self, arguments, split=None):
        super().__init__(arguments, split)
        self.json_folder = self.params.ext_file
        os.makedirs(self.json_folder, exist_ok=True)

    def save_info_to_json(self, index):
        file_path = os.path.join(self.json_folder, f"info_{index}.json")
        with open(file_path, 'w') as f:
            json.dump(self.info, f)

    def SaveRowToJSON(self, index):
        self.ConvertforExternalSaving(self.data.loc[index], index)
        self.save_info_to_json(index)

    def SaveAllToJSON(self):
        for index in tqdm(self.data.index.tolist(), total=len(self.data.index.tolist()), desc="Saving to JSON"):
            self.SaveRowToJSON(index)

    def LoadJSONDataFrame(self, folder_path):
        try:
            data = []
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".json"):
                    file_path = os.path.join(folder_path, file_name)
                    with open(file_path, 'r') as f:
                        data.append(json.load(f))
            self.data = pd.DataFrame(data)
        except Exception as e:
            print(f"Failed to load DataFrame from JSON folder {folder_path}")
            print(traceback.print_exc())

    def GetAllIndices(self):
        try:
            indices = self.data["Indices"].unique().tolist()
            return indices
        except Exception as e:
            print(f"Failed to retrieve indices from DataFrame")
            print(traceback.print_exc())
            return []

    def load_json_as_info_dict(self, folder_path, index):
        try:
            file_path = os.path.join(folder_path, f"info_{index}.json")
            with open(file_path, 'r') as f:
                self.info = json.load(f)
        except Exception as e:
            print(f"Failed to load info from JSON file {file_path} for index {index}")
            print(traceback.print_exc())

    def GetInfo(self, index):
        self.load_json_as_info_dict(self.json_folder, index)

    def SampleJSON(self, index):
        self.GetInfo(index)
        self.creategraphsample()
        return self.sample

    def FingerprintModelSamplerJSON(self, index):
        self.GetInfo(index)
        self.creategraphsample()
        return self.sample


class GraphJSONDataset(JSONCommands):
    def __init__(self, arguments, split=None):
        super().__init__(arguments, split)
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
                print(self.root + '--' + str(index) + ' failed')
                print(traceback.print_exc())
                return None


class FingerprintJSONDataset(JSONCommands):
    def __init__(self, arguments, split=None):
        super().__init__(arguments, split)
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
                print(self.root + '--' + str(index) + ' failed')
                print(traceback.print_exc())
                return None


class JSONSaver(JSONCommands):
    def __init__(self, arguments, split=None):
        super().__init__(arguments, split)
        self.SaveAllToJSON()
