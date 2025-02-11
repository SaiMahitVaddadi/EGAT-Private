import dgl
from .utils import filter_none



class ReactionCollator:

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
            names,types,Rgraphs, Pgraphs, smiles,targets = map(list, zip(*self.samples))
            Rbatched_graph = dgl.batch(Rgraphs)
            Pbatched_graph = dgl.batch(Pgraphs)
            return names,types,Rbatched_graph, Pbatched_graph, smiles,targets
        except ValueError:
            return None
    
    def additionals(self,samples):
        self.samples = filter_none(samples)
        try:
            names,types,Rgraphs, Pgraphs, smiles,targets,additionals = map(list, zip(*self.samples))
            Rbatched_graph = dgl.batch(Rgraphs)
            Pbatched_graph = dgl.batch(Pgraphs)
            return names,types,Rbatched_graph, Pbatched_graph, smiles,targets,additionals
        except ValueError:
            return None
        
    def addons(self,samples):
        self.samples = filter_none(samples)
        try:
            names,types,Rgraphs, Pgraphs, smiles,targets,Radd,Padd = map(list, zip(*self.samples))
            Rbatched_graph = dgl.batch(Rgraphs)
            Pbatched_graph = dgl.batch(Pgraphs)
            return names,types,Rbatched_graph, Pbatched_graph, smiles,targets,Radd,Padd
        except ValueError:
            return None
    
    def allprops(self,samples):
        self.samples = filter_none(samples)
        try:
            names,types,Rgraphs, Pgraphs, smiles,targets,additionals,Radd,Padd = map(list, zip(*self.samples))
            Rbatched_graph = dgl.batch(Rgraphs)
            Pbatched_graph = dgl.batch(Pgraphs)
            return names,types,Rbatched_graph, Pbatched_graph,smiles,targets,additionals,Radd,Padd
        except ValueError:
            return None
