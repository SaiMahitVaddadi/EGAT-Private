from rdkit import Chem
from mordred import Calculator, descriptors
from descriptastorus.descriptors import rdNormalizedDescriptors,rdDescriptors
from molskill.scorer import MolSkillScorer
from RAscore import RAscore_NN #For tensorflow and keras based models
from RAscore import RAscore_XGB #For XGB based models
import QEPPI as ppi
from rdkit import Chem
from syba.syba import SybaClassifier
from .rdkit import RDKitDescriptors

class Descriptors:
    def __init__(self,calc='rdkit',is3d=False):
        self.calc = calc
        self.is3d = is3d
        if self.calc == 'descriptasourus-normed':
            self.model = rdNormalizedDescriptors.RDKit2DNormalized()
        elif self.calc == 'descriptasourus':
            self.model = rdDescriptors.RDKit2D()
        elif self.calc == 'molskill':
            self.model = MolSkillScorer()
        elif self.calc == 'rascore-nn':
            self.model = RAscore_NN.RAScorerNN() 
        elif self.calc == 'rascore-xgb':
            self.model = RAscore_XGB.RAScorerXGB()
        elif self.calc == 'qep':
            q = ppi.QEPPI_Calculator()
            q.read()
        elif self.calc == 'syba':
            self.model = SybaClassifier()
            self.model.fitDefaultScore()
        

    def MolSkill(self, smis):
        return self.model.score(smis) 

    def QEP(self, smis):
        q = ppi.QEPPI_Calculator()
        q.read()
        self.Load(smis)
        return [q.qeppi(mol) for mol in self.mols]

    def RAScoreNN(self, smis):
        self.Load(smis)
        return [self.model.predict(mol) for mol in self.mols]

    def RAScoreXGB(self, smis):
        self.Load(smis)
        return [self.model.predict(mol) for mol in self.mols]

    def Syba(self, smis):
        syba = SybaClassifier()
        syba.fitDefaultScore()
        self.Load(smis)
        return [syba.predict(mol) for mol in self.mols]

    def Load(self, smi:str):
        self.mol = Chem.MolFromSmiles(smi)
        self.mol = Chem.AddHs(self.mol)
        if self.is3d:
            Chem.EmbedMolecule(self.mol)
            Chem.UFFOptimizeMolecule(self.mol)
        self.mols = [self.mol]
    
    def Compute(self,smi):
        if self.calc == 'descriptasourus-normed':
            return self.model.process(smi)
        elif self.calc == 'descriptasourus':
            return self.model.process(smi)
        elif self.calc == 'molskill':
            return self.MolSkill(smi)
        elif self.calc == 'rascore-nn':
            return self.RAScoreNN(smi)
        elif self.calc == 'rascore-xgb':
            return self.RAScoreXGB(smi)
        elif self.calc == 'qep':
            return self.QEP(smi)
        elif self.calc == 'syba':
            return self.Syba(smi)

    