from ..base import BaseFeaturizer
from dataclasses import dataclass



@dataclass
class NeighborParams:
    neighbor: str  # Options: 'onlyH', 'onlyCHNO', 'onlyorganic', 'all'

class NeighborInformation(BaseFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
        
    def GetNeighbors(self,ind):
        neighbors = [self.matrixdescriptors.element[counti] for counti,i in enumerate(self.matrixdescriptors.adj_mat[ind,:]) if i != 0]
        if self.params.neighbor == 'onlyH':
            NE_count = [neighbors.count('H')]
        elif self.params.neighbor == 'onlyCHNO':
            NE_count = [neighbors.count('H'), neighbors.count('C'), neighbors.count('N'), neighbors.count('O')]
        elif self.params.neighbor == 'onlyorganic':
            NE_count = [neighbors.count('H'), neighbors.count('C'), neighbors.count('N'), neighbors.count('O'), 
                        neighbors.count('P'), neighbors.count('S'), neighbors.count('F'), neighbors.count('Cl'), 
                        neighbors.count('Br'), neighbors.count('I')]
        elif self.params.neighbor == 'all':
            NE_count = [neighbors.count(element) for element in self.properties.element_encode.keys()]
        else:
            NE_count = []
        return NE_count
    
    def GetNeighborsChemprop(self,ind):
        pass