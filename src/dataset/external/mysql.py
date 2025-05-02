import mysql.connector
import traceback
from tqdm import tqdm
from commands import ExternalSaveCommands
from ..base.commands import DatasetCommands
import pandas as pd

class SQLCommands(ExternalSaveCommands):
    def __init__(self, arguments):
        super().__init__(arguments)

    def setupsqlserver(self):
        conn = mysql.connector.connect(
            host=self.params.dbhost,
            user=self.params.dbuser,
            password=self.params.dbpassword,
            database=self.params.dbname
        )
        cursor = conn.cursor()
        return conn, cursor

    def setinfointosql(self, index, conn, cursor):
        table_name = f"info_{index}"
        columns = self.info.keys()
        values = [self.info[col] for col in columns]
        create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} (" + ", ".join([f"{col} TEXT" for col in columns]) + ")"
        cursor.execute(create_table_query)
        insert_query = f"INSERT INTO {table_name} (" + ", ".join(columns) + ") VALUES (" + ", ".join(["%s"] * len(values)) + ")"
        cursor.execute(insert_query, values)
        conn.commit()
        cursor.close()
        conn.close()

    def setinfolistintosql(self, index, conn, cursor):
        table_name = f"info_{index}"
        columns = ['all_result']
        values = [(item,) for item in self.infolist]
        create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} (" + ", ".join([f"{col} TEXT" for col in columns]) + ")"
        cursor.execute(create_table_query)
        insert_query = f"INSERT INTO {table_name} (" + ", ".join(columns) + ") VALUES (" + ", ".join(["%s"] * len(columns)) + ")"
        cursor.executemany(insert_query, values)
        conn.commit()
        cursor.close()
        conn.close()

    def setinfointosqlasseries(self, index, conn, cursor):
        for i in range(len(self.infolist)):
            table_name = f"info_{index}_{i}"
            columns = self.infolist[i].keys()
            values = [self.infolist[i][col] for col in columns]
            create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} (" + ", ".join([f"{col} TEXT" for col in columns]) + ")"
            cursor.execute(create_table_query)
            insert_query = f"INSERT INTO {table_name} (" + ", ".join(columns) + ") VALUES (" + ", ".join(["%s"] * len(values)) + ")"
            cursor.execute(insert_query, values)
            conn.commit()
        cursor.close()
        conn.close()

    def SaveRowToSQLite(self, index):
        try:
            infolistusage = self.OrganizeData(index)
            conn, cursor = self.setupsqlserver()

            if infolistusage:
                if self.params.saveconfsseparately:
                    self.setinfointosqlasseries(index, conn, cursor)
                else:
                    self.setinfolistintosql(index, conn, cursor)
            else:
                self.setinfointosql(index, conn, cursor)
        except Exception as e:
            self.__ExternalException(index)

    def SaveInfoToSQLite(self):
        for index in tqdm(self.data.index.tolist(), total=len(self.data.index.tolist()), desc="Saving to MySQL"):
            self.SaveRowToSQLite(index)

    def LoadSQLDataFrame(self):
        try:
            conn = mysql.connector.connect(
                host=self.params.dbhost,
                user=self.params.dbuser,
                password=self.params.dbpassword,
                database=self.params.dbname
            )
            query = "SELECT * FROM data"
            self.data = pd.read_sql_query(query, conn)
            conn.close()
        except Exception as e:
            print(f"Failed to load SQL DataFrame from {self.params.dbname}")
            print(traceback.print_exc())

    def GetAllIndices(self):
        try:
            conn = mysql.connector.connect(
                host=self.params.dbhost,
                user=self.params.dbuser,
                password=self.params.dbpassword,
                database=self.params.dbname
            )
            cursor = conn.cursor()
            query = "SELECT DISTINCT Indices FROM data"
            cursor.execute(query)
            indices = [row[0] for row in cursor.fetchall()]
            conn.close()
            return indices
        except Exception as e:
            print(f"Failed to retrieve indices from {self.params.dbname}")
            print(traceback.print_exc())
            return []

    def loadsqlasinfodict(self, index, i=None):
        try:
            conn = mysql.connector.connect(
                host=self.params.dbhost,
                user=self.params.dbuser,
                password=self.params.dbpassword,
                database=self.params.dbname
            )
            cursor = conn.cursor(dictionary=True)
            if i is None:
                table_name = f"info_{index}"
            else:
                table_name = f"info_{index}_{i}"
            query = f"SELECT * FROM {table_name}"
            cursor.execute(query)
            rows = cursor.fetchall()
            self.info = {col: row[col] for row in rows for col in row}
            conn.close()
        except Exception as e:
            print(f"Failed to load info from {self.params.dbname}")
            print(traceback.print_exc())

    def loadsqlasinfolist(self, index):
        try:
            conn = mysql.connector.connect(
                host=self.params.dbhost,
                user=self.params.dbuser,
                password=self.params.dbpassword,
                database=self.params.dbname
            )
            cursor = conn.cursor()
            table_name = f"info_{index}"
            query = f"SELECT * FROM {table_name}"
            cursor.execute(query)
            rows = cursor.fetchall()
            self.infolist = [row[0] for row in rows]
            conn.close()
        except Exception as e:
            print(f"Failed to load info list from {self.params.dbname}")
            print(traceback.print_exc())

    def GetInfo(self, index):
        infolistusage = self.__useinfolist()
        if infolistusage:
            if self.params.saveconfsseparately:
                self.loadsqlasinfodict(index)
            else:
                self.loadsqlasinfolist(index)
        else:
            self.loadsqlasinfodict(index)

    def SampleSQL(self, index):
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

    def FingerprintModelSamplerSQL(self, index):
        infolistusage = self.__useinfolist()
        self.GetInfo(index)
        self.samples = []
        self.samples += self._sampleindices(infolistusage)
        self.samples += self._samplereactiontype(infolistusage)
        samples += self._sampletargets(infolistusage)
        samples += self._sampleadditionals(infolistusage)
        samples += self.Addons()
        return samples


class GraphSQLDataset(SQLCommands):
    def __init__(self, arguments):
        super().__init__(arguments)
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
    def __init__(self, arguments):
        super().__init__(arguments)
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
    def __init__(self, arguments):
        super().__init__(arguments)
        self.SaveInfoToSQLite()
