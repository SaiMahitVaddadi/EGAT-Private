from .randomwalk import RandomWalk
from rdkit import Chem
from rdkit.Chem import DataStructs
import networkx as nx
import numpy as np
from ....utils.descriptors.egat.functional import FunctionalGroups
from scipy.stats import skew, kurtosis
from scipy.linalg import pinv
import random
from dataclasses import dataclass

@dataclass
class GlobalAtomParams:
    getclusteringcoeff: bool = False
    getclosenesscentrality: bool = False
    getdegreecentrality: bool = False
    getgraphdensity: bool = False
    getavgdegneighbors: bool = False
    getnodeclustering: bool = False
    getassortativity: bool = False
    getspectralradius: bool = False
    getdegreeentropy: bool = False
    getlocalbertzct: bool = False
    getcoulomb: bool = False
    getpagerank: bool = False

class GlobalAtomInformation(RandomWalk):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)


    def GetClusteringCoefficient(self,node):
        if self.params.getclusteringcoeff:
            return [self.ClusteringCoefficient(node)]
        else:
            return [] 
        
    def GetClosenessCentrality(self, node):
        if self.params.getclosenesscentrality:
            return [self.ClosenessCentrality(node)]
        else:
            return []
        
    def GetDegreeCentrality(self, node):
        if self.params.getdegreecentrality:
            return [self.DegreeCentrality(node)]
        else:
            return []
        
    def GetDensity(self,ind):
        if self.params.getgraphdensity:
            return [self.Density(ind)]
        else:
            return []
    
    def GetAvgDegreeNeighbors(self,ind):
        if self.params.getavgdegneighbors:
            return [self.AvgDegreeNeighbors(ind)]
        else:
            return []
        

    def GetNodeClustering(self,ind):
        if self.params.getnodeclustering:
            return [self.Clustering(ind)]
        else:
            return []
    
    def GetAssortativity(self,ind):
        if self.params.getassortativity:
            from icecream import ic
            ic(self.Assortativity(ind))
            return [self.Assortativity(ind)]
        else:
            return []
    
    def GetSpectralRadius(self,ind):
        if self.params.getspectralradius:
            return [self.SpectralRadius(ind)]
        else:
            return []
    
    def GetDegreeEntropy(self,ind):
        if self.params.getdegreeentropy:
            return [self.DegreeEntropy(ind)]
        else:
            return []
        
    def GetLocalBertzCT(self,ind):
        if self.params.getlocalbertzct:
            return [self.LocalBertzCT(ind)]
        else:
            return []
        
    def GetCoulombValue(self, atom_index):
        if self.params.getcoulomb: 
            """
            Calculate the Coulomb value for a given atom in the molecule.
            """
            
            positions = list(self.positions.values())
            charges = [atom.GetAtomicNum() for atom in self.conformer.GetAtoms()]
            
            coulomb_value = 0.0
            for i, pos_i in enumerate(positions):
                if i == atom_index:
                    continue
                distance = np.linalg.norm(np.array(positions[atom_index]) - np.array(pos_i))
                if distance > 0:
                    coulomb_value += charges[atom_index] * charges[i] / distance
            
            return [coulomb_value]
        else:
            return []
    

    def GetCoulombValueForBond(self, edge):
        if self.params.getcoulomb: 
            """
            Calculate the Coulomb value for a given bond in the molecule.
            """
            
            positions = list(self.positions.values())
            charges = [atom.GetAtomicNum() for atom in self.conformer.GetAtoms()]
            
            atom1_index, atom2_index = edge
            
            distance = np.linalg.norm(np.array(positions[atom1_index]) - np.array(positions[atom2_index]))
            if distance > 0:
                coulomb_value = charges[atom1_index] * charges[atom2_index] / distance
            else:
                coulomb_value = 0.0
            
            return [coulomb_value]
        else:
            return []

    def GetPageRank(self, ind):
        if self.params.getpagerank:
            """
            Calculate the PageRank value for a given atom or bond in the molecule.
            """
            pagerank = nx.pagerank(self.G)
            
            if isinstance(ind, int):  # Atom (node)
                return [pagerank.get(ind, 0.0)]
            else:
                return [0]
        else:
            return []
    
    
