
from .hdf import HDF5Saver,HDF5Params
from .json import JSONSaver,JSONParams
from .mongo import MongoSaver,MongoParams
from .mysql import SQLSaver,SQLParams
from .postgres import PostgresQLSaver,PostgresQLParams
from .spark import SparkSaver,SparkParams
from .sqlite import SQLiteSaver,SQLiteParams

class GraphGenParams(HDF5Params,JSONParams,MongoParams,SQLParams,PostgresQLParams,SparkParams,SQLiteParams):
    pass


class GraphGeneration:
    def __init__(self,params):
        self.params = params
    
    def grabsaver(self):
        self.saver = globals()[f'{self.params.dataset}Saver']

    def save(self):
        self.grabsaver()
        self.saver()
        

