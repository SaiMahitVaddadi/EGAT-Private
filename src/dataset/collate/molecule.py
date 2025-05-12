import dgl
from .utils import filter_none
from .expansions import Expansions,molecularexpandfcnforadditionals,molecularexpandfcnfortargets,molecularexpandfcnforaddons,molecularexpandfcnforallprops
from .basecollate import BaseCollator
#Write a function that expands the Batches of the dataset by the graph. 

class MolecularCollator(BaseCollator):

    """
    Function that takes the dataset and gets the batched data. 
    Parameters
    ----------
    samples: torch.DataLoader

    Returns
    -------
    names: string
            Names of the .json files being looked at
    types: string
            Names of the reaction type of the .json file
    Rbatched_graph: DGL graph
            Graph of the reactant
    smiles: string
            Smiles of the Reactant and Product
    targets: float
            Target values
    Radd: list
            List of the RDKit Global Features passed from the Reactant
    additionals: float
            Values being passed in the FFNN
    """     
    def __init__(self, mode="targets",multicomp=False):
        super().__init__(mode=mode,multicomp=multicomp)

    def targets(self,samples):
        names,types,Rgraphs, smiles,targets = self.runmapping(samples)
        if self.useexpansion(Rgraphs): names,types,Rbatched_graph, smiles,targets = molecularexpandfcnfortargets(names,types,Rgraphs, smiles,targets,multicomp=self.multicomp)
        else: Rbatched_graph = dgl.batch(Rgraphs)
        return names,types,Rbatched_graph, smiles,targets
        
    def additionals(self,samples):
        names,types,Rgraphs, smiles,targets,additionals = self.runmapping(samples)
        if self.useexpansion(Rgraphs): names,types,Rbatched_graph, smiles,targets,additionals = molecularexpandfcnforadditionals(names,types,Rgraphs, smiles,targets,additionals,multicomp=self.multicomp)
        else: Rbatched_graph = dgl.batch(Rgraphs)
        return names,types,Rbatched_graph, smiles,targets,additionals
        
    def addons(self,samples):
        names,types,Rgraphs, smiles,targets,Radd = self.runmapping(samples)
        if self.useexpansion(Rgraphs): names,types,Rbatched_graph, smiles,targets,Radd = molecularexpandfcnforaddons(names,types,Rgraphs, smiles,targets,Radd,multicomp=self.multicomp)
        else: Rbatched_graph = dgl.batch(Rgraphs)
        return names,types,Rbatched_graph, smiles,targets,Radd
        
    def allprops(self,samples):
        names,types,Rgraphs, smiles,targets,additionals,Radd = self.runmapping(samples)
        if self.useexpansion(Rgraphs): names,types,Rbatched_graph, smiles,targets,additionals,Radd = molecularexpandfcnforallprops(names,types,Rgraphs, smiles,targets,additionals,Radd,multicomp=self.multicomp)
        else: Rbatched_graph = dgl.batch(Rgraphs)
        return names,types,Rbatched_graph, smiles,targets,additionals,Radd
        
    def __call__(self, samples):
        return self.callfcn(samples)

class MolecularFingerprintCollator(BaseCollator):

    """
    Function that takes the dataset and gets the batched data. 
    Parameters
    ----------
    samples: torch.DataLoader

    Returns
    -------
    names: string
            Names of the .json files being looked at
    types: string
            Names of the reaction type of the .json file
    Rbatched_graph: DGL graph
            Graph of the reactant
    smiles: string
            Smiles of the Reactant and Product
    targets: float
            Target values
    Radd: list
            List of the RDKit Global Features passed from the Reactant
    additionals: float
            Values being passed in the FFNN
    """    

    def __init__(self, mode="targets",multicomp=False):
        super().__init__(mode=mode,multicomp=multicomp)
        
    def targets(self,samples):
        self.samples = filter_none(samples)
        names,types,smiles,targets = map(list, zip(*self.samples))
        return names,types,smiles,targets
        
    def additionals(self,samples):
        self.samples = filter_none(samples)
        names,types,smiles,targets,additionals = map(list, zip(*self.samples))
        return names,types,smiles,targets,additionals
        
    def addons(self,samples):
        self.samples = filter_none(samples)
        names,types,smiles,targets,Radd,Padd = map(list, zip(*self.samples))
        return names,types,smiles,targets,Radd,Padd
        
    def allprops(self,samples):
        names,types,smiles,targets,additionals,Radd,Padd = map(list, zip(*self.samples))
        return names,types,smiles,targets,additionals,Radd,Padd
    
    def __call__(self, samples):
        return self.callfcn(samples)
        


    