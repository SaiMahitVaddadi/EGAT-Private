from ...utils.descriptors.egat.encodings import Encodings
from ...utils.descriptors.egat.reactive import Reactive
from ...utils.descriptors.egat.molmatdesc import MolMatDesc
from ...utils.descriptors.egat.radicals import RDKElectronInfo,YARPElectronInfo
from ...utils.descriptors.egat.stereo import StereoChemistry
from ...utils.matrices.graph_seps import graph_seps
from ...utils.misc.taffi_functions import return_rings,adjmat_to_adjlist
from ...utils.descriptors.geometry.geometry import ConformerGenerator
from rdkit import Chem
from dataclasses import dataclass
from typing import Literal
from .base import BaseFeaturizer


class GeomFeaturizer(BaseFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
    
    def GenerateConformers(self):
        # Fix for each 3D Geometry Tools That's There
        self.conformers = ConformerGenerator(self.matrixdescriptors,self.params.conformer.nconfs,self.params.conformer.method,
                                             self.params.conformer.seed,self.params.conformer.verbose,self.params.conformer.rdkoptimizer)

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
