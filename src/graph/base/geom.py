from ...utils.descriptors.geometry.geometry import ConformerGeneratorEGAT
from rdkit import Chem
from .base import BaseFeaturizer


class GeomFeaturizer(BaseFeaturizer,object):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
    
    def GenerateConformers(self):
        # Fix for each 3D Geometry Tools That's There
        self.conformers = ConformerGeneratorEGAT(self.matrixdescriptors,self.params.conformer)
        try:
            confgenerator = getattr(self.conformers,f'Generatewith{self.params.conformer.generator}',None)
        except:
            self.params.conformer.generator = 'RDKit'
            confgenerator = getattr(self.conformers,f'Generatewith{self.params.conformer.generator}',None)

        if self.params.conformer.generator == 'RDKit':
            confgenerator(self.params.conformer.seed)
        elif self.params.conformer.generator == 'Auto3D':
            confgenerator(self.params.conformer.seed)
        elif self.params.conformer.generator == 'ASE':
            confgenerator(self.params.conformer.seed)
        elif self.params.conformer.generator == 'CREST':
            confgenerator(self.params.conformer.seed)
        elif self.params.conformer.generator == 'CREGEN':
            confgenerator(self.params.conformer.seed)
        elif self.params.conformer.generator == 'QCG':
            confgenerator(self.params.conformer.seed)
        elif self.params.conformer.generator == 'OpenFF':
            confgenerator(self.params.conformer.seed)
        elif self.params.conformer.generator == 'OpenMM':
            confgenerator(self.params.conformer.seed)
        elif self.params.conformer.generator == 'TorsDiff':
            confgenerator(self.params.conformer.seed)
        
        self.matrixdescriptors.new_mol = self.conformers.egatecule.new_mol
    
    def Spread(self, num_conformers):
        mol = self.conformers.egatecule.new_mol
        mols = [Chem.Mol(mol, confId=i) for i in range(num_conformers)]
        self.matrixdescriptors.new_mol = mols

    