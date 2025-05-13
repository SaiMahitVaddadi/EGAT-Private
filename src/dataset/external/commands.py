from ..base.commands import DatasetCommands
import sqlite3
import traceback
from tqdm import tqdm

class ExternalSaveCommands(DatasetCommands):
    def __init__(self, arguments,split=None):
        super().__init__(arguments,split)
    
    def savesplittoinfo(self,rxn):
        if 'split' in self.data.columns:
            self.info['split'] = rxn['split']
        else:
            self.info['split'] = 'all'
    
    def deleteanygraphs(self):
        keys_to_remove = [key for key in self.info.keys() if 'Graph' in key]
        for key in keys_to_remove:
            del self.info[key]
    
    def ConvertforExternalSaving(self,rxn,index):
        self.Convert(index)
        self.savesplittoinfo(rxn)
        self.deleteanygraphs()

    def __ExternalException(self,index):
        print(self.root + '--'+ str(index) + ' failed')
        print(traceback.print_exc())
