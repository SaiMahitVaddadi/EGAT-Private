
from .base.commands import DatasetCommands
import traceback
class GraphDataset(DatasetCommands):
    def __init__(self,arguments,split=None):
        super().__init__(arguments,split)
    
    def __ItemException(self,index):
        print(self.root + '--'+ str(index) + ' failed')
        print(traceback.print_exc())
        return None
                
    def __getitem__(self, index):
        if index in self.cache:
            samples = self.cache[index]
            return samples
        else:
            try:
                samples = self.Sample(index)
                self.node_feats_length = self.node_feature_length
                self.edge_feats_length = self.bond_feature_length
                self.addonlength = self.addonlength
                if len(self.cache) < self.cache_size:
                    self.cache[index] = samples
                return samples
            except Exception as e:
                self.__ItemException(index)
    
    def __len__(self):
        return len(self.data)
                
class FingerprintDataset(DatasetCommands):
    def __init__(self,arguments,split=None):
        super().__init__(arguments,split)
    
    def __ItemException(self,index):
        print(self.root + '--'+ str(index) + ' failed')
        print(traceback.print_exc())
        return None
    
    def __getitem__(self, index):
        if index in self.cache:
            samples = self.cache[index]
            return samples
        else:
            try:
                samples = self.FingerprintModelSampler(index)
                if len(self.cache) < self.cache_size:
                    self.cache[index] = samples
                return samples
            except Exception as e:
                self.__ItemException(index)
    
    def __len__(self):
        return len(self.data)