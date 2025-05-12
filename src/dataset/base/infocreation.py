from .helpers.featurizer import FeauritizerFunctions
from .helpers.inchissmilesandtypes import InchisSMILESandTypes
from .helpers.normfunctions import NormFunctions
from .helpers.featstoinfo import FeaturestoInfo
from .helpers.dglgraphcreation import DGLGraphCreation
from .helpers.addons import Addons

class InfoCreation(NormFunctions,InchisSMILESandTypes,FeauritizerFunctions,FeaturestoInfo,DGLGraphCreation,Addons):
    def __init__(self, params,Rind):
        """
        This class is responsible for creating and managing information related to molecular and reaction data.

        Attributes:
            - params: Configuration parameters for the info creation process.
            - info: A dictionary to store various information related to molecules and reactions.
            - featurizer: A dictionary to store featurizers for different components (e.g., molecules, reactions).
            - infolist: A list to store additional information or features.
            - rxntype_columns: A list to store columns related to reaction types.
        """
        self.params = params
        self.info = {}
        self.info['Indices'] = Rind


    def GetRow(self,Rind):
        if isinstance(Rind,str):
            rxn = self.data.loc[Rind,:]
        else:
            rxn = self.data.iloc[Rind,:]
        return rxn
    
    def AddIndex(self,Rind):
        self.info['Indices'] = Rind
        
    def createinfodictforgraph(self,Rind):
        rxn = self.GetRow(Rind)
        self.AddIndex(Rind)
        self.AddSMILESandInchIsandTypes(rxn)
        self.AddTargets(rxn)
        self.AddAdditionals(rxn)
        self.AddAddons()
        if self.params.fingerprint == False: 
            self.ChooseFeaturizer(rxn)
            self.AddFeatures()
            self.DGLSetup()
            self.AddGraphs()

    def run(self):
        self.createinfodictforgraph()
        

    
    
    

