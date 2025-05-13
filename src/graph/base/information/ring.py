from ..base import BaseFeaturizer
from dataclasses import dataclass



@dataclass
class RingParams:
    removeringinfo: bool = False
    removearomaticity: bool = False

class RingInformation(BaseFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
        
    def RingCheck(self,ind):
        if not self.params.removeringinfo:
            if True in [ind in ra_list for ra_list in self.ring_atoms]: inR = 1
            else: inR = 0
            return [inR]
        else:
            return []
    
    def AromaticityCheck(self,ind,ind_=0):
        if not self.params.removearomaticity:
            ###### GET AROMATICITY
            if self.matrixdescriptors.element[ind_] == 'H': aromaticity = 0
            elif self.stereo.atom_aromatic[ind]: aromaticity = 1
            else: aromaticity = 0
            return [aromaticity]
        else:
            return []
    
    def BondinRing(self,edge):
        if not self.params.removeringinfo:
            if True in [(edge[0] in ra_list and edge[1] in ra_list) for ra_list in self.ring_atoms]: inR = 1
            else: inR = 0
            return [inR]
        else:
            return []
