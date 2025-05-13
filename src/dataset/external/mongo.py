from pymongo import MongoClient
import traceback
from tqdm import tqdm
from .commands import ExternalSaveCommands
from ..base.commands import DatasetCommands
import pandas as pd
from dataclasses import dataclass


@dataclass
class MongoParams:
    mongo_uri: str = "mongodb://localhost:27017"
    database_name: str = "default_db"
    collection_name: str = "default_collection"

class MongoDBCommands(ExternalSaveCommands):
    def __init__(self, arguments, split=None):
        super().__init__(arguments, split)
        self.mongo_uri = self.params.mongo_uri
        self.database_name = self.params.database_name
        self.collection_name = self.params.collection_name
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[self.database_name]
        self.collection = self.db[self.collection_name]

    def save_info_to_mongo(self, index):
        document = {"index": index, "info": self.info}
        self.collection.insert_one(document)

    def SaveRowToMongo(self, index):
        self.ConvertforExternalSaving(self.data.loc[index], index)
        self.save_info_to_mongo(index)

    def SaveAllToMongo(self):
        for index in tqdm(self.data.index.tolist(), total=len(self.data.index.tolist()), desc="Saving to MongoDB"):
            self.SaveRowToMongo(index)

    def LoadMongoDataFrame(self):
        try:
            cursor = self.collection.find()
            self.data = pd.DataFrame(list(cursor))
        except Exception as e:
            print(f"Failed to load DataFrame from MongoDB collection {self.collection_name}")
            print(traceback.print_exc())

    def GetAllIndices(self):
        try:
            indices = self.collection.distinct("index")
            return indices
        except Exception as e:
            print(f"Failed to retrieve indices from MongoDB collection {self.collection_name}")
            print(traceback.print_exc())
            return []

    def loadmongoasinfodict(self, index):
        try:
            document = self.collection.find_one({"index": index})
            if document:
                self.info = document["info"]
            else:
                raise ValueError(f"No document found for index {index}")
        except Exception as e:
            print(f"Failed to load info from MongoDB collection {self.collection_name} for index {index}")
            print(traceback.print_exc())

    def GetInfo(self, index):
        self.loadmongoasinfodict(index)

    def SampleMongo(self, index):
        self.GetInfo(index)
        self.creategraphsample()
        return self.sample

    def FingerprintModelSamplerMongo(self, index):
        self.GetInfo(index)
        self.creategraphsample()
        return self.sample


class GraphMongoDataset(MongoDBCommands):
    def __init__(self, arguments, split=None):
        super().__init__(arguments, split)
        self.GetAllIndices()

    def __getitem__(self, index):
        if index in self.cache:
            samples = self.cache[index]
        else:
            try:
                samples = self.SampleMongo(index)
                if len(self.cache) < self.cache_size:
                    self.cache[index] = samples
            except Exception as e:
                print(self.root + '--' + str(index) + ' failed')
                print(traceback.print_exc())
                return None


class FingerprintMongoDataset(MongoDBCommands):
    def __init__(self, arguments, split=None):
        super().__init__(arguments, split)
        self.GetAllIndices()

    def __getitem__(self, index):
        if index in self.cache:
            samples = self.cache[index]
        else:
            try:
                samples = self.FingerprintModelSamplerMongo(index)
                if len(self.cache) < self.cache_size:
                    self.cache[index] = samples
            except Exception as e:
                print(self.root + '--' + str(index) + ' failed')
                print(traceback.print_exc())
                return None


class MongoSaver(MongoDBCommands):
    def __init__(self, arguments, split=None):
        super().__init__(arguments, split)
        self.SaveAllToMongo()