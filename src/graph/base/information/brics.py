from ..base import BaseFeaturizer
from rdkit.Chem import BRICS
from dataclasses import dataclass


@dataclass
class BRICSParams:
    checkbricsbond: bool = False


class BRICSInformation(BaseFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
        
    def _convert(self,ind):
        atom = self.matrixdescriptors.new_mol.GetAtoms()[ind]
        return atom.GetAtomMapNum()

    def GetBRICSDecomposition(self):
        brics_bonds = list(BRICS.FindBRICSBonds(self.matrixdescriptors.new_mol))
        bond_breaks = [(self._convert(bond[0][0]), self._convert(bond[0][1])) for bond in brics_bonds]
        return bond_breaks
    
    def IsPartOfBRICSBond(self, ind):
        self.brics = self.GetBRICSDecomposition()
        if self.params.checkbricsbond:
            for bond in self.brics:
                if ind in bond:
                    return [1]
            return [0]
        else:
            return []
        
    def IsBRICSBond(self,edge):
        self.brics = self.GetBRICSDecomposition()
        if self.params.checkbricsbond:
            if edge in self.brics: return [0,1]
            else: return [1,0]
        else:
            return []

    