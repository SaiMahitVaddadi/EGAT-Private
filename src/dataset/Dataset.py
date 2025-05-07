
from base.commands import DatasetCommands

class GraphDataset(DatasetCommands):
    def __init__(self,arguments,split=None):
        super().__init__(arguments,split)
                
    def __getitem__(self, index):
        if index in self.cache:
            samples = self.cache[index]
        else:
            try:
                samples = self.Sample(index)
                if len(self.cache) < self.cache_size:
                    self.cache[index] = samples
            except Exception as e:
                self.__ItemException(index)
                
class FingerprintDataset(DatasetCommands):
    def __init__(self,arguments,split=None):
        super().__init__(arguments,split)
    
    def __getitem__(self, index):
        if index in self.cache:
            samples = self.cache[index]
        else:
            try:
                samples = self.FingerprintModelSampler(index)
                if len(self.cache) < self.cache_size:
                    self.cache[index] = samples
            except Exception as e:
                self.__ItemException(index)
                