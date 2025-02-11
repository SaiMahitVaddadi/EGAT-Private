import random
from typing import List, Tuple, Optional
from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    MolFromSmiles,
    MolToSmiles,
    MolStandardize,
    MolToInchiKey,
    SDWriter,
    MACCSkeys,
    RDKFingerprint,
    LayeredFingerprint
)
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.AtomPairs.Pairs import GetAtomPairFingerprint
from rdkit.Chem.EState.Fingerprinter import FingerprintMol
from rdkit.Chem.AtomPairs.Torsions import GetTopologicalTorsionFingerprintAsIntVect
from rdkit.Chem.Pharm2D import Generate
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D
from rdkit.Chem.rdmolops import RenumberAtoms
from rdkit.DataStructs.cDataStructs import UIntSparseIntVect
from molvs import Standardizer
from typing import List
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.Chem.Pharm2D import Generate
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D
from rdkit.Chem.Pharm3D import EmbedLib, Pharmacophore
from skfp.fingerprints import *
from rdkit.Chem.rdMolDescriptors import BCUT2D
import numpy as np
from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
from mapchiral.mapchiral import encode, jaccard_similarity
from mxfp.mxfp import MXFPCalculator
from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer
from selfies import encoder as selfies_encoder
from molfeat.trans import MoleculeTransformer
from molfeat.calc.pharmacophore import Pharmacophore2D
from molfeat.trans.pretrained import PretrainedDGLTransformer
from transformers import AutoModel,AutoTokenizer


class Fingerprint:
    def __init__(self):
        pass
    
    def smiles_to_fps(self, smiles: List[str],fpname:str,variant:str = 'raw_bits',radius:int=3,use_counts:bool=True,use_features:bool=True,size:int=2048,fuzziness:float = .5,paths:int=5,author:str = '',model:str = '',padding:bool=True) -> np.array:
        """
        Converts a list of SMILES strings to their corresponding EState fingerprints.
        :param smiles: List of SMILES strings.
        :return: List of EState fingerprints as UIntSparseIntVect.
        """
        mols = self.smiles_to_mols(smiles)
        valid_mols = [mol for mol in mols if mol is not None]
        fps = []
        for smile in smiles:
            mol = MolFromSmiles(smile)
            if mol is None:
                fps.append(None)
            else:
                fps.append(self.fpfunc(mol,fpname=fpname,variant=variant,radius=radius,use_counts=use_counts,use_features=use_features,size=size,fuzziness=fuzziness,paths=paths,author=author,model=model,padding=padding))
        return fps

    def fpfunc(self,mol,fpname:str='ecfp',variant:str = 'raw_bits',radius:int=3,use_counts:bool=True,use_features:bool=True,size:int=2048,fuzziness:float = .5,paths:int=5,author:str = '',model:str = '',padding:bool=True):
        if fpname == 'ecfp':
            return AllChem.GetMorganFingerprint(mol, radius, useCounts=use_counts, useFeatures=use_features)
        elif fpname == 'bcut':
            return BCUT2D(mol)
        if fpname == 'rdkit':
            return RDKFingerprint(mol, maxPath=int(variant))
        elif fpname == 'maccs':
            return MACCSkeys.GenMACCSKeys(mol)
        elif fpname == 'pubchem':
            return GetAtomPairFingerprint(mol)
        elif fpname == 'layered':
            return LayeredFingerprint(mol,maxPath=int(variant))
        elif fpname == 'estate':
            return FingerprintMol(mol)[0]
        elif fpname == 'torsional':
            return GetTopologicalTorsionFingerprintAsIntVect(mol)
        elif fpname == 'avalon':
            return GetAvalonFP(mol)
        elif fpname == '2dpharm':
            factory = Gobbi_Pharm2D.factory
            return Generate.Gen2DFingerprint(mol, factory)
        elif fpname == 'mapchiral':
            return encode(mol, max_radius=radius, n_permutations=size, mapping=False)
        elif fpname == 'mxfp':
            calculator = MXFPCalculator()
            return calculator.mxfp_from_mol(mol)
        elif fpname == 'autocorr':
            fp = AutocorrFingerprint()
            smiles = Chem.MolToSmiles(mol)
            return fp.transform(smiles)
        elif fpname == 'erg':
            fp = ERGFingerprint(fuzz_increment=fuzziness,max_path=paths)
            smiles = Chem.MolToSmiles(mol)
            return fp.transform(smiles)
        elif fpname == 'crippen':
            fp = GhoseCrippenFingerprint(count=use_counts)
            smiles = Chem.MolToSmiles(mol)
            return fp.transform(smiles)
        elif fpname == 'roth':
            fp = KlekotaRothFingerprint(count=use_counts)
            smiles = Chem.MolToSmiles(mol)
            return fp.transform(smiles)
        elif fpname == 'laggner':
            fp = LaggnerFingerprint(count=use_counts)
            smiles = Chem.MolToSmiles(mol)
            return fp.transform(smiles)
        elif fpname == 'lingo':
            fp = LingoFingerprint(substring_length=int(variant), count=use_counts)
            smiles = Chem.MolToSmiles(mol)
            return fp.transform(smiles)
        elif fpname == 'map4':
            fp = MAPFingerprint(radius=radius, variant=variant)
            smiles = Chem.MolToSmiles(mol)
            return fp.transform(smiles)
        elif fpname == 'mhfp':
            fp = MHFPFingerprint(radius=radius, min_radius=int(variant), variant=variant)
            smiles = Chem.MolToSmiles(mol)
            return fp.transform(smiles)
        elif fpname == 'morse':
            fp = MORSEFingerprint()
            smiles = Chem.MolToSmiles(mol)
            return fp.transform(smiles)
        elif fpname == 'mordred':
            fp = MordredFingerprint()
            smiles = Chem.MolToSmiles(mol)
            return fp.transform(smiles)
        elif fpname == 'mqns':
            fp = MQNsFingerprint(count=use_counts)
            smiles = Chem.MolToSmiles(mol)
            return fp.transform(smiles)
        elif fpname == 'patternfp':
            fp = PatternFingerprint()
            smiles = Chem.MolToSmiles(mol)
            return fp.transform(smiles)
        elif fpname == 'physchemfp':
            fp = PhysiochemicalPropertiesFingerprint(variant=variant, count=use_counts)
            smiles = Chem.MolToSmiles(mol)
            return fp.transform(smiles)
        elif fpname == 'secfp':
            fp = SECFPFingerprint(radius=radius, min_radius=int(variant))
            smiles = Chem.MolToSmiles(mol)
            return fp.transform(smiles)
        elif fpname == 'vsa':
            fp = VSAFingerprint(variant=variant)
            smiles = Chem.MolToSmiles(mol)
            return fp.transform(smiles)
        elif fpname == 'whim':
            fp = WHIMFingerprint()
            smiles = Chem.MolToSmiles(mol)
            return fp.transform(smiles)
    
        elif fpname == 'cdkstandard':
            smiles = Chem.MolToSmiles(mol)
            return self.PyFingerprintUserFunction(smiles, "standard")
        elif fpname == 'cdkextended':
            smiles = Chem.MolToSmiles(mol)
            return self.PyFingerprintUserFunction(smiles, "extended")
        elif fpname == 'cdkgraph':
            smiles = Chem.MolToSmiles(mol)
            return self.PyFingerprintUserFunction(smiles, "graph")
        elif fpname == 'hybridization':
            smiles = Chem.MolToSmiles(mol)
            return self.PyFingerprintUserFunction(smiles, "hybridization")
        elif fpname == 'cdkshortestpath':
            smiles = Chem.MolToSmiles(mol)
            return self.PyFingerprintUserFunction(smiles, "shortestpath")
        elif fpname == 'cdksubstructures':
            smiles = Chem.MolToSmiles(mol)
            return self.PyFingerprintUserFunction(smiles, "cdk-substructure")
        elif fpname == 'circular':
            smiles = Chem.MolToSmiles(mol)
            return self.PyFingerprintUserFunction(smiles, "circular")
        elif fpname == 'cdkatompairs':
            smiles = Chem.MolToSmiles(mol)
            return self.PyFingerprintUserFunction(smiles, "cdk-atompairs")
        elif fpname == 'babelfp':
            smiles = Chem.MolToSmiles(mol)
            if radius == 2:
                return self.PyFingerprintUserFunction(smiles, "fp2")
            elif radius == 3:
                return self.PyFingerprintUserFunction(smiles, "fp3")
            elif radius == 4:
                return self.PyFingerprintUserFunction(smiles, "fp4")
        elif fpname == 'spectrophore':
            smiles = Chem.MolToSmiles(mol)
            return self.PyFingerprintUserFunction(smiles, "spectrophore")
        elif fpname == 'mol2vec':
            smiles = Chem.MolToSmiles(mol)
            return self.PyFingerprintUserFunction(smiles, "mol2vec")
        elif fpname == 'chemgpt12b':
            transformer = PretrainedHFTransformer(kind='ChemGPT-1.2B', notation='selfies', dtype=float)
            smiles = selfies_encoder(Chem.MolToSmiles(mol))
            return transformer([smiles])
        elif fpname == 'chemgpt19m':
            transformer = PretrainedHFTransformer(kind='ChemGPT-19M', notation='selfies', dtype=float)
            smiles = selfies_encoder(Chem.MolToSmiles(mol))
            return transformer([smiles])
        elif fpname == 'chemgpt47m':
            transformer = PretrainedHFTransformer(kind='ChemGPT-4.7M', notation='selfies', dtype=float)
            smiles = selfies_encoder(Chem.MolToSmiles(mol))
            return transformer([smiles])
        elif fpname == 'molt5':
            transformer = PretrainedHFTransformer(kind='MolT5', notation='smiles', dtype=float)
            smiles = Chem.MolToSmiles(mol)
            return transformer([smiles])
        elif fpname == 'robertazincllm':
            transformer = PretrainedHFTransformer(kind='Roberta-Zinc480M-102M', notation='smiles', dtype=float)
            smiles = Chem.MolToSmiles(mol)
            return transformer([smiles])
        elif fpname == 'chembertmtrllm':
            transformer = PretrainedHFTransformer(kind='ChemBERTa-77M-MTR', notation='smiles', dtype=float)
            smiles = Chem.MolToSmiles(mol)
            return transformer([smiles])
        elif fpname == 'chembertmlmllm':
            transformer = PretrainedHFTransformer(kind='ChemBERTa-77M-MLM', notation='smiles', dtype=float)
            smiles = Chem.MolToSmiles(mol)
            return transformer([smiles])
        elif fpname == 'gpt2llm':
            transformer = PretrainedHFTransformer(kind='GPT2-Zinc480M-87M', notation='smiles', dtype=float)
            smiles = Chem.MolToSmiles(mol)
            return transformer([smiles])
        elif fpname == 'scaffoldkeys':
            transformer = MoleculeTransformer(featurizer='scaffoldkeys', dtype=float)
            smiles = Chem.MolToSmiles(mol)
            return transformer([smiles])
        elif fpname == 'cats':
            transformer = MoleculeTransformer(featurizer='cats2d', dtype=float)
            smiles = Chem.MolToSmiles(mol)
            return transformer([smiles])
        elif fpname == 'catsrdkit':
            transformer = MoleculeTransformer(featurizer=Pharmacophore2D(factory='cats'), dtype=float)
            smiles = Chem.MolToSmiles(mol)
            return transformer([smiles])
        elif fpname == 'gobbipharm':
            transformer = MoleculeTransformer(featurizer=Pharmacophore2D(factory='gobbi'), dtype=float)
            smiles = Chem.MolToSmiles(mol)
            return transformer([smiles])
        elif fpname == 'pmapper':
            transformer = MoleculeTransformer(featurizer=Pharmacophore2D(factory='pmapper'), dtype=float)
            smiles = Chem.MolToSmiles(mol)
            return transformer([smiles])
        elif fpname == 'chemblgin':
            transformer = PretrainedDGLTransformer(kind='gin_supervised_masking', dtype=float)
            smiles = Chem.MolToSmiles(mol)
            return transformer([smiles])
        elif fpname == 'jtvaezinc':
            transformer = PretrainedDGLTransformer(kind='jtvae_zinc_no_kl', dtype=float)
            smiles = Chem.MolToSmiles(mol)
            return transformer([smiles])
        elif fpname == 'chemblginedgepred':
            transformer = PretrainedDGLTransformer(kind='gin_supervised_edgepred', dtype=float)
            smiles = Chem.MolToSmiles(mol)
            return transformer([smiles])
        elif fpname == 'chemblgininfomax':
            transformer = PretrainedDGLTransformer(kind='gin_supervised_infomax', dtype=float)
            smiles = Chem.MolToSmiles(mol)
            return transformer([smiles])
        elif fpname == 'graphormerpcqm':
            transformer = GraphormerTransformer(kind='pcqm4mv2_graphormer_base', dtype=float)
            smiles = Chem.MolToSmiles(mol)
            return transformer([smiles])
        elif fpname == 'chemblgincontext':
            transformer = PretrainedDGLTransformer(kind='gin_supervised_contextpred', dtype=float)
            smiles = Chem.MolToSmiles(mol)
            return transformer([smiles])
        elif fpname == 'hftransformer':
            checkpoint = f'{author}/{model}'
            smiles = Chem.MolToSmiles(mol)
            if 'MoLFormer' in model:
                tokenizer = AutoTokenizer.from_pretrained(checkpoint,trust_remote_code=True)
                return tokenizer(smi,padding=padding,return_tensors='pt')
            else:
                tokenizer = AutoTokenizer.from_pretrained(checkpoint)
                if 'selfies' in model:
                    smiles = self.smiles_to_selfies(smiles)
                return tokenizer(smiles,return_tensors='pt')
    
    def PyFingerprintUserFunction(self,smiles:List[str],function:str) -> List[UIntSparseIntVect]:
        fps = [fp.to_numpy() for fp in get_fingerprints(smiles,function)]
        return np.array(fps)
    
    
    def smiles_to_selfies(self, smiles: List[str]) -> List[str]:
        """
        Converts a list of SMILES strings to their corresponding SELFIES strings.
        :param smiles: List of SMILES strings.
        :return: List of SELFIES strings.
        """
        selfies_list = [selfies_encoder(smile) for smile in smiles]
        return selfies_list
