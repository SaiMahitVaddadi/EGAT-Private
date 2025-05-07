from pymongo import MongoClient
import traceback
from tqdm import tqdm
from commands import ExternalSaveCommands
from ..base.commands import DatasetCommands
import pandas as pd

class MongoCommands(ExternalSaveCommands):
    def __init__(self, arguments,split=None):
        super().__init__(arguments,split)

    def setup_mongo_client(self):
        client = MongoClient(self.params.dbhost, self.params.dbport)
        db = client[self.params.dbname]
        return client, db

    def set_info_into_mongo(self, index, db):
        collection_name = f"info_{index}"
        collection = db[collection_name]
        collection.insert_one(self.info)

    def set_info_list_into_mongo(self, index, db):
        collection_name = f"info_{index}"
        collection = db[collection_name]
        documents = [{"all_result": item} for item in self.infolist]
        collection.insert_many(documents)

    def set_info_into_mongo_as_series(self, index, db):
        for i, info in enumerate(self.infolist):
            collection_name = f"info_{index}_{i}"
            collection = db[collection_name]
            collection.insert_one(info)

    def SaveRowToMongo(self, index):
        try:
            infolistusage = self.OrganizeData(index)
            client, db = self.setup_mongo_client()

            if infolistusage:
                if self.params.saveconfsseparately:
                    self.set_info_into_mongo_as_series(index, db)
                else:
                    self.set_info_list_into_mongo(index, db)
            else:
                self.set_info_into_mongo(index, db)
            client.close()
        except Exception as e:
            self.__ExternalException(index)

    def SaveInfoToMongo(self):
        for index in tqdm(self.data.index.tolist(), total=len(self.data.index.tolist()), desc="Saving to MongoDB"):
            self.SaveRowToMongo(index)

    def LoadMongoDataFrame(self):
        try:
            client, db = self.setup_mongo_client()
            collection = db["data"]
            data = list(collection.find())
            self.data = pd.DataFrame(data)
            client.close()
        except Exception as e:
            print(f"Failed to load MongoDB DataFrame from {self.params.dbname}")
            print(traceback.print_exc())

    def GetAllIndices(self):
        try:
            client, db = self.setup_mongo_client()
            collection = db["data"]
            indices = collection.distinct("Indices")
            client.close()
            return indices
        except Exception as e:
            print(f"Failed to retrieve indices from {self.params.dbname}")
            print(traceback.print_exc())
            return []

    def load_mongo_as_info_dict(self, index, i=None):
        try:
            client, db = self.setup_mongo_client()
            if i is None:
                collection_name = f"info_{index}"
            else:
                collection_name = f"info_{index}_{i}"
            collection = db[collection_name]
            self.info = collection.find_one()
            client.close()
        except Exception as e:
            print(f"Failed to load info from {self.params.dbname}")
            print(traceback.print_exc())

    def load_mongo_as_info_list(self, index):
        try:
            client, db = self.setup_mongo_client()
            collection_name = f"info_{index}"
            collection = db[collection_name]
            self.infolist = [doc["all_result"] for doc in collection.find()]
            client.close()
        except Exception as e:
            print(f"Failed to load info list from {self.params.dbname}")
            print(traceback.print_exc())

    def GetInfo(self, index):
        infolistusage = self.__useinfolist()
        if infolistusage:
            if self.params.saveconfsseparately:
                self.load_mongo_as_info_dict(index)
            else:
                self.load_mongo_as_info_list(index)
        else:
            self.load_mongo_as_info_dict(index)

    def SampleMongo(self, index):
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

    def FingerprintModelSamplerMongo(self, index):
        infolistusage = self.__useinfolist()
        self.GetInfo(index)
        self.samples = []
        self.samples += self._sampleindices(infolistusage)
        self.samples += self._samplereactiontype(infolistusage)
        samples += self._sampletargets(infolistusage)
        samples += self._sampleadditionals(infolistusage)
        samples += self.Addons()
        return samples


class GraphMongoDataset(MongoCommands):
    def __init__(self, arguments,split=None):
        super().__init__(arguments,split)
        self.LoadMongoDataFrame()
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


class FingerprintMongoDataset(MongoCommands):
    def __init__(self, arguments,split=None):
        super().__init__(arguments,split)
        self.LoadMongoDataFrame()
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


class MongoSaver(MongoCommands):
    def __init__(self, arguments,split=None):
        super().__init__(arguments,split)
        self.SaveInfoToMongo()
