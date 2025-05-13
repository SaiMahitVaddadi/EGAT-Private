from .dataset_setups import DatasetSetups
from .infocreation import InfoCreation
from .dicttosample import InfoToSample
from icecream import ic

class DatasetCommands(DatasetSetups,InfoCreation,InfoToSample):
    def __init__(self, arguments,split=None):
        self.params = arguments
        self.split = split
        self.SetupInits()
        self.checkmolecular()
        self.InitialSetup() 

    def Convert(self,Rind):
        self.createinfodictforgraph(Rind)
    
    def Sample(self,Rind):
        self.Convert(Rind)
        self.creategraphsample()
        return self.sample

    def FingerprintModelSampler(self,Rind):
        self.Convert(Rind)
        self.creategraphsample()
        return self.sample

