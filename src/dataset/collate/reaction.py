import dgl
from .utils import filter_none
from .basecollate import BaseCollator
from .expansions import Expansions,reactionexpandfcnforadditionals,reactionexpandfcnfortargets,reactionexpandfcnforaddons,reactionexpandfcnforallprops

class ReactionCollator(BaseCollator):

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
    Pbatched_graph: DGL graph
            Graph of the Product
    smiles: string
            Smiles of the Reactant and Product
    targets: float
            Target values
    Radd: list
            List of the RDKit Global Features passed from the Reactant
    Padd: list
            List of the RDKit Global Features passed from the Product
    additionals: float
            Values being passed in the FFNN
    """    

    

    def targets(self,samples):
        names,types,Rgraphs,Pgraphs, smiles,targets = self.runmapping(samples)
        if self.useexpansion(Rgraphs): names,types,Rbatched_graph,Pbatched_graph, smiles,targets = reactionexpandfcnfortargets(names,types,Rgraphs,Pgraphs, smiles,targets,multicomp=self.multicomp)
        else: 
            Rbatched_graph = dgl.batch(Rgraphs)
            Pbatched_graph = dgl.batch(Pgraphs)
        return names,types,Rbatched_graph,Pbatched_graph, smiles,targets
        
    def additionals(self,samples):
        names,types,Rgraphs,Pgraphs, smiles,targets,additionals = self.runmapping(samples)
        if self.useexpansion(Rgraphs): names,types,Rbatched_graph,Pbatched_graph, smiles,targets,additionals = reactionexpandfcnforadditionals(names,types,Rgraphs,Pgraphs, smiles,targets,additionals,multicomp=self.multicomp)
        else: 
            Rbatched_graph = dgl.batch(Rgraphs)
            Pbatched_graph = dgl.batch(Pgraphs)
        return names,types,Rbatched_graph,Pbatched_graph, smiles,targets,additionals
        
    def addons(self,samples):
        names,types,Rgraphs,Pgraphs, smiles,targets,Radd = self.runmapping(samples)
        if self.useexpansion(Rgraphs): names,types,Rbatched_graph,Pbatched_graph, smiles,targets,Radd = reactionexpandfcnforaddons(names,types,Rgraphs,Pgraphs, smiles,targets,Radd,multicomp=self.multicomp)
        else: 
            Rbatched_graph = dgl.batch(Rgraphs)
            Pbatched_graph = dgl.batch(Pgraphs)
        return names,types,Rbatched_graph,Pbatched_graph, smiles,targets,Radd
        
    def allprops(self,samples):
        names,types,Rgraphs,Pgraphs, smiles,targets,additionals,Radd = self.runmapping(samples)
        if self.useexpansion(Rgraphs): names,types,Rbatched_graph,Pbatched_graph, smiles,targets,additionals,Radd = reactionexpandfcnforallprops(names,types,Rgraphs,Pgraphs, smiles,targets,additionals,Radd,multicomp=self.multicomp)
        else: 
            Rbatched_graph = dgl.batch(Rgraphs)
            Pbatched_graph = dgl.batch(Pgraphs)
        return names,types,Rbatched_graph,Pbatched_graph, smiles,targets,additionals,Radd
class ReactionFingerprintCollator:

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
    Pbatched_graph: DGL graph
            Graph of the Product
    smiles: string
            Smiles of the Reactant and Product
    targets: float
            Target values
    Radd: list
            List of the RDKit Global Features passed from the Reactant
    Padd: list
            List of the RDKit Global Features passed from the Product
    additionals: float
            Values being passed in the FFNN
    """    

    

    def targets(self,samples):
        self.samples = filter_none(samples)
        try:
            names,types,smiles,targets = map(list, zip(*self.samples))
            return names,types, smiles,targets
        except ValueError:
            return None
    
    def additionals(self,samples):
        self.samples = filter_none(samples)
        try:
            names,types,smiles,targets,additionals = map(list, zip(*self.samples))
            return names,types, smiles,targets,additionals
        except ValueError:
            return None
        
    def addons(self,samples):
        self.samples = filter_none(samples)
        try:
            names,types,smiles,targets,Radd,Padd = map(list, zip(*self.samples))
            return names,types, smiles,targets,Radd,Padd
        except ValueError:
            return None
    
    def allprops(self,samples):
        self.samples = filter_none(samples)
        try:
            names,types,smiles,targets,additionals,Radd,Padd = map(list, zip(*self.samples))
            return names,types,smiles,targets,additionals,Radd,Padd
        except ValueError:
            return None
