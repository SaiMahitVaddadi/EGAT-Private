import dgl
from .utils import filter_none
class BaseCollator:
    def __init__(self, mode="targets",multicomp=False):
        self.multicomp = multicomp
        self.mode = mode

    def runmapping(self,samples):
        self.samples = filter_none(samples)
        return map(list, zip(*self.samples))

    def useexpansion(self,Rgraphs):
        return any(isinstance(Rgraph, list) for Rgraph in Rgraphs)
    
    def callfcn(self,samples):
        if self.mode == "targets":
            return self.targets(samples)
        elif self.mode == "additionals":
            return self.additionals(samples)
        elif self.mode == "addons":
            return self.addons(samples)
        elif self.mode == "allprops":
            return self.allprops(samples)
        else:
            raise ValueError(f"Unknown collate mode: {self.mode}")