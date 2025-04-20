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