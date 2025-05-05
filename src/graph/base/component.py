from .base import BaseFeaturizer
from ...utils.descriptors.egat.encodings import Encodings
from ...utils.descriptors.egat.reactive import Reactive
from rdkit import Chem
from rxnmapper import RXNMapper



class BaseReactionComponentFeaturizer(BaseFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
        self.ReactiveGraphSepMat()
        if not self.IsAtomMapped(): self.CreateAtomMapping()
        else: self.am_smiles = self.smiles
        
    def IsAtomMapped(self):
        molecule = Chem.MolFromSmiles(self.smiles)
        for atom in molecule.GetAtoms():
            if atom.GetAtomMapNum() != 0:
                return True
        return False
    
    