import os
import traceback
import pandas as pd
from tqdm import tqdm
from .commands import ExternalSaveCommands
from ..base.commands import DatasetCommands
import sqlite3
from dataclasses import dataclass

@dataclass
class SQLiteParams:
    ext_file: str = ''
    cache_size: int = 100  # Default cache size
    root: str = ""         # Default root directory

class SQLiteCommands(ExternalSaveCommands):
    def __init__(self, arguments, split=None):
        super().__init__(arguments, split)
        self.sqlite_file = self.params.ext_file
        os.makedirs(os.path.dirname(self.sqlite_file), exist_ok=True)
        self.setup_sqlite()

    def setup_sqlite(self):
        self.conn = sqlite3.connect(self.sqlite_file)
        self.cursor = self.conn.cursor()

    def save_info_to_sqlite(self, index):
        table_name = f"info_{index}"
        columns = self.info.keys()
        values = [self.info[col] for col in columns]
        create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} (" + ", ".join([f"{col} TEXT" for col in columns]) + ")"
        self.cursor.execute(create_table_query)
        insert_query = f"INSERT INTO {table_name} (" + ", ".join(columns) + ") VALUES (" + ", ".join(["?"] * len(values)) + ")"
        self.cursor.execute(insert_query, values)
        self.conn.commit()

    def SaveRowToSQLite(self, index):
        self.ConvertforExternalSaving(self.data.loc[index], index)
        self.save_info_to_sqlite(index)

    def SaveAllToSQLite(self):
        for index in tqdm(self.data.index.tolist(), total=len(self.data.index.tolist()), desc="Saving to SQLite"):
            self.SaveRowToSQLite(index)

    def LoadSQLiteDataFrame(self, file_path):
        try:
            self.conn = sqlite3.connect(file_path)
            query = "SELECT * FROM data"
            self.data = pd.read_sql_query(query, self.conn)
        except Exception as e:
            print(f"Failed to load DataFrame from SQLite file {file_path}")
            print(traceback.print_exc())

    def GetAllIndices(self):
        try:
            query = "SELECT DISTINCT Indices FROM data"
            self.cursor.execute(query)
            indices = [row[0] for row in self.cursor.fetchall()]
            return indices
        except Exception as e:
            print(f"Failed to retrieve indices from SQLite database")
            print(traceback.print_exc())
            return []

    def load_sqlite_as_info_dict(self, file_path, index):
        try:
            table_name = f"info_{index}"
            query = f"SELECT * FROM {table_name}"
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            self.info = {col[0]: row for col, row in zip(self.cursor.description, rows)}
        except Exception as e:
            print(f"Failed to load info from SQLite file {file_path} for index {index}")
            print(traceback.print_exc())

    def GetInfo(self, index):
        self.load_sqlite_as_info_dict(self.sqlite_file, index)

    def SampleSQLite(self, index):
        self.GetInfo(index)
        self.creategraphsample()
        return self.sample

    def FingerprintModelSamplerSQLite(self, index):
        self.GetInfo(index)
        self.creategraphsample()
        return self.sample


class GraphSQLiteDataset(SQLiteCommands):
    def __init__(self, arguments, split=None):
        super().__init__(arguments, split)
        self.GetAllIndices()

    def __getitem__(self, index):
        if index in self.cache:
            samples = self.cache[index]
        else:
            try:
                samples = self.SampleSQLite(index)
                if len(self.cache) < self.cache_size:
                    self.cache[index] = samples
            except Exception as e:
                print(self.root + '--' + str(index) + ' failed')
                print(traceback.print_exc())
                return None


class FingerprintSQLiteDataset(SQLiteCommands):
    def __init__(self, arguments, split=None):
        super().__init__(arguments, split)
        self.GetAllIndices()

    def __getitem__(self, index):
        if index in self.cache:
            samples = self.cache[index]
        else:
            try:
                samples = self.FingerprintModelSamplerSQLite(index)
                if len(self.cache) < self.cache_size:
                    self.cache[index] = samples
            except Exception as e:
                print(self.root + '--' + str(index) + ' failed')
                print(traceback.print_exc())
                return None


class SQLiteSaver(SQLiteCommands):
    def __init__(self, arguments, split=None):
        super().__init__(arguments, split)
        self.SaveAllToSQLite()
