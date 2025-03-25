from rdkit import Chem
from mordred import Calculator, descriptors
from fsscore.score import Scorer
from fsscore.models.ranknet import LitRankNet
from fsscore.utils.paths import PRETRAIN_MODEL_PATH
from descriptastorus.descriptors import rdNormalizedDescriptors,rdDescriptors
from molskill.scorer import MolSkillScorer
from RAscore import RAscore_NN #For tensorflow and keras based models
from RAscore import RAscore_XGB #For XGB based models
import QEPPI as ppi
from rdkit import Chem
from syba.syba import SybaClassifier
from .rdkit import RDKitDescriptors
from sklearn.preprocessing import StandardScaler

class Descriptors:
    def __init__(self,calc='rdkit',is3d=False):
        self.calc = calc
        self.is3d = is3d

        if self.calc == 'fsscore':
            self.model = LitRankNet.load_from_checkpoint(PRETRAIN_MODEL_PATH) # does not work on CPU
        elif self.calc == 'descriptasourus-normed':
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
            syba = SybaClassifier()
            syba.fitDefaultScore()
        elif self.calc == 'rdkit':
            self.model = RDKitDescriptors()

    def FSScore(self,smi):
        scorer = Scorer(model=self.model)
        return scorer.score(smi)

    def MolSkill(self, smis):
        return self.model.score(smis) 
    def QEP(self, smis):
        q = ppi.QEPPI_Calculator()
        q.read()
        self.LoadMany(smis)
        return [q.qeppi(mol) for mol in self.mols]

    def RAScoreNN(self, smis):
        self.LoadMany(smis)
        return [self.model.predict(mol) for mol in self.mols]

    def RAScoreXGB(self, smis):
        self.LoadMany(smis)
        return [self.model.predict(mol) for mol in self.mols]

    def Syba(self, smis):
        syba = SybaClassifier()
        syba.fitDefaultScore()
        self.LoadMany(smis)
        return [syba.predict(mol) for mol in self.mols]

    def Rawr(self, smis):
        features = []
        for smi in smis:
            if self.model.process(smi)[0] == True:
                features.append(self.model.process(smi)[1:])
            else:
                features.append(None)
        return features


    def Load(self, smi:str):
        self.mol = Chem.MolFromSmiles(smi)
        self.mol = Chem.AddHs(self.mol)
        if self.is3d:
            Chem.EmbedMolecule(self.mol)
            Chem.UFFOptimizeMolecule(self.mol)

    def LoadMany(self,smis:list):
        self.mols = []
        for smi in smis:
            self.Load(smi)
            self.mols.append(self.mol)

    
    def LoadMordred(self):
        self.calc = Calculator(descriptors, ignore_3D=self.is3d)
    
    def ComputeForOne(self):
        return self.calc(self.mol)

    def ComputeForMany(self):
        return self.calc.pandas(self.mols)
    
    def Mordred(self,smis,descs=None):
        self.LoadMordred()
        self.LoadMany(smis)
        feats = self.ComputeForMany()
        feats = feats.loc[descs]
        return feats
    
    def RDKit(self,smis,descs=None):
        self.LoadMany(smis)
        if descs is None:
            descs = list(self.model.descriptors.keys())
        return [self.model.compute(mol, desc) for mol in self.mols for desc in descs]
    
    
