import mysql.connector
import traceback
from tqdm import tqdm
from .commands import ExternalSaveCommands
from ..base.commands import DatasetCommands
import pandas as pd
from dataclasses import dataclass

@dataclass
class SQLParams:
    dbhost: str = "localhost"
    dbuser: str = "root"
    dbpassword: str = ""
    dbname: str = "test"

    
class SQLCommands(ExternalSaveCommands):
    def __init__(self, arguments, split=None):
        super().__init__(arguments, split)
        self.dbhost = self.params.dbhost
        self.dbuser = self.params.dbuser
        self.dbpassword = self.params.dbpassword
        self.dbname = self.params.dbname

    def setup_connection(self):
        conn = mysql.connector.connect(
            host=self.dbhost,
            user=self.dbuser,
            password=self.dbpassword,
            database=self.dbname
        )
        return conn

    def save_info_to_sql(self, index):
        conn = self.setup_connection()
        cursor = conn.cursor()
        table_name = f"info_{index}"
        df = pd.DataFrame([self.info])
        columns = df.columns.tolist()
        create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} (" + ", ".join([f"{col} TEXT" for col in columns]) + ")"
        cursor.execute(create_table_query)
        insert_query = f"INSERT INTO {table_name} (" + ", ".join(columns) + ") VALUES (" + ", ".join(["%s"] * len(columns)) + ")"
        cursor.execute(insert_query, df.iloc[0].tolist())
        conn.commit()
        cursor.close()
        conn.close()

    def SaveRowToSQL(self, index):
        self.ConvertforExternalSaving(self.data.loc[index], index)
        self.save_info_to_sql(index)

    def SaveAllToSQL(self):
        for index in tqdm(self.data.index.tolist(), total=len(self.data.index.tolist()), desc="Saving to MySQL"):
            self.SaveRowToSQL(index)

    def LoadSQLDataFrame(self):
        try:
            conn = self.setup_connection()
            query = "SELECT * FROM data"
            self.data = pd.read_sql_query(query, conn)
            conn.close()
        except Exception as e:
            print(f"Failed to load DataFrame from MySQL database {self.dbname}")
            print(traceback.print_exc())

    def GetAllIndices(self):
        try:
            conn = self.setup_connection()
            cursor = conn.cursor()
            query = "SELECT DISTINCT Indices FROM data"
            cursor.execute(query)
            indices = [row[0] for row in cursor.fetchall()]
            conn.close()
            return indices
        except Exception as e:
            print(f"Failed to retrieve indices from MySQL database {self.dbname}")
            print(traceback.print_exc())
            return []

    def load_sql_as_info_dict(self, index):
        try:
            conn = self.setup_connection()
            cursor = conn.cursor(dictionary=True)
            table_name = f"info_{index}"
            query = f"SELECT * FROM {table_name}"
            cursor.execute(query)
            rows = cursor.fetchall()
            self.info = {col: row[col] for row in rows for col in row}
            conn.close()
        except Exception as e:
            print(f"Failed to load info from MySQL database {self.dbname} for index {index}")
            print(traceback.print_exc())

    def GetInfo(self, index):
        self.load_sql_as_info_dict(index)

    def SampleSQL(self, index):
        self.GetInfo(index)
        self.creategraphsample()
        return self.sample

    def FingerprintModelSamplerSQL(self, index):
        self.GetInfo(index)
        self.creategraphsample()
        return self.sample


class GraphSQLDataset(SQLCommands):
    def __init__(self, arguments, split=None):
        super().__init__(arguments, split)
        self.LoadSQLDataFrame()
        self.GetAllIndices()

    def __getitem__(self, index):
        if index in self.cache:
            samples = self.cache[index]
        else:
            try:
                samples = self.SampleSQL(index)
                if len(self.cache) < self.cache_size:
                    self.cache[index] = samples
            except Exception as e:
                print(self.root + '--' + str(index) + ' failed')
                print(traceback.print_exc())
                return None


class FingerprintSQLDataset(SQLCommands):
    def __init__(self, arguments, split=None):
        super().__init__(arguments, split)
        self.LoadSQLDataFrame()
        self.GetAllIndices()

    def __getitem__(self, index):
        if index in self.cache:
            samples = self.cache[index]
        else:
            try:
                samples = self.FingerprintModelSamplerSQL(index)
                if len(self.cache) < self.cache_size:
                    self.cache[index] = samples
            except Exception as e:
                print(self.root + '--' + str(index) + ' failed')
                print(traceback.print_exc())
                return None


class SQLSaver(SQLCommands):
    def __init__(self, arguments, split=None):
        super().__init__(arguments, split)
        self.SaveAllToSQL()
