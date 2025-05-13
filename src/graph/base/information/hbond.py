from ..base import BaseFeaturizer
from dataclasses import dataclass


@dataclass
class HydrogenBondParams:
    check_hbond: bool

class HydrogenBondInformation(BaseFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
    
    def HydrogenBondCheck(self,ind):
        if self.params.check_hbond:
            hbond_info = []
            if self.matrixdescriptors.element[ind] in ['O', 'N', 'F', 'Cl', 'Br', 'I']:
                hbond_info = [1, 0,0]  # Donor
            elif self.matrixdescriptors.element[ind] == 'H':
                for neighbor in range(len(self.matrixdescriptors.element)):
                    if self.matrixdescriptors.adj_mat[ind][neighbor] > 0 and self.matrixdescriptors.element[neighbor] in ['O', 'N', 'F', 'Cl', 'Br', 'I']:
                        hbond_info = [0, 1,0]  # Acceptor
                    break
            else:
                hbond_info = [0, 0,1]  # Neither
            return hbond_info
        else:
            return []
