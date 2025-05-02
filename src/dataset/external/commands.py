from ..base.commands import DatasetCommands
import sqlite3
import traceback
from tqdm import tqdm

class ExternalSaveCommands(DatasetCommands):
    def __init__(self, arguments):
        super().__init__(arguments)
    
    def savesplittoinfo(self,rxn):
        if 'split' in self.data.columns:
            self.info['split'] = rxn['split']
        else:
            self.info['split'] = 'all'
    
    def savesplittoinfolist(self,rxn):
        if 'split' in self.data.columns:
            for info in self.infolist:
                if 'split' in info:
                    info['split'] = rxn['split']
                else:
                    info['split'] = 'all'

    def AddSplit(self, rxn,infolistusage=False):
        if infolistusage: 
            self.savesplittoinfolist(rxn)
        else:
            self.savesplittoinfo(rxn)
    
    def OrganizeData(self,index):
        infolistusage = self.__useinfolist()
        self.InfoorInfoList(index)
        self.AddSplit(self.data.iloc[index],infolistusage)
        return infolistusage
    
    def __ExternalException(self,index):
        print(self.root + '--'+ str(index) + ' failed')
        print(traceback.print_exc())
