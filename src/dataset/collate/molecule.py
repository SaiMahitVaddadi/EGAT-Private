import dgl
from .utils import filter_none


class MolecularCollator:

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

    


    def targets(self,samples):
        self.samples = filter_none(samples)
        try:
            names,types,Rgraphs, smiles,targets = map(list, zip(*self.samples))
            Rbatched_graph = dgl.batch(Rgraphs)
            return names,types,Rbatched_graph, smiles,targets
        except:
            return None

    def additionals(self,samples):
        self.samples = filter_none(samples)
        try:
            names,types,Rgraphs, smiles,targets,additionals = map(list, zip(*self.samples))
            Rbatched_graph = dgl.batch(Rgraphs)
            return names,types,Rbatched_graph, smiles,targets,additionals
        except ValueError:
            return None

    def addons(self,samples):
        self.samples = filter_none(samples)
        try:
            names,types,Rgraphs, smiles,targets,Radd,Padd = map(list, zip(*self.samples))
            Rbatched_graph = dgl.batch(Rgraphs)
            return names,types,Rbatched_graph, smiles,targets,Radd,Padd
        except ValueError:
            return None
        
    def allprops(self,samples):
        self.samples = filter_none(samples)
        try:
            names,types,Rgraphs, smiles,targets,additionals,Radd,Padd = map(list, zip(*self.samples))
            Rbatched_graph = dgl.batch(Rgraphs)
            return names,types,Rbatched_graph, smiles,targets,additionals,Radd,Padd
        except ValueError:
            return None
        

class MolecularFingerprintCollator:

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

    


    def targets(self,samples):
        self.samples = filter_none(samples)
        try:
            names,types,smiles,targets = map(list, zip(*self.samples))
            return names,types,smiles,targets
        except:
            return None

    def additionals(self,samples):
        self.samples = filter_none(samples)
        try:
            names,types,smiles,targets,additionals = map(list, zip(*self.samples))
            return names,types,smiles,targets,additionals
        except ValueError:
            return None

    def addons(self,samples):
        self.samples = filter_none(samples)
        try:
            names,types,smiles,targets,Radd,Padd = map(list, zip(*self.samples))
            return names,types,smiles,targets,Radd,Padd
        except ValueError:
            return None
        
    def allprops(self,samples):
        try:
            names,types,smiles,targets,additionals,Radd,Padd = map(list, zip(*self.samples))
            return names,types,smiles,targets,additionals,Radd,Padd
        except ValueError:
            return None
        


    