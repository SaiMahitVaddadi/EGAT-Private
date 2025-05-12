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
from rdkit.Chem.Pharm2D import SigFactory
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
from mapchiral.mapchiral import encode, jaccard_similarity
from mxfp.mxfp import MXFPCalculator
from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer
from selfies import encoder as selfies_encoder
from molfeat.trans import MoleculeTransformer
from molfeat.calc.pharmacophore import Pharmacophore2D
from molfeat.trans.pretrained import PretrainedDGLTransformer
from transformers import AutoModel,AutoTokenizer


from rdkit import DataStructs
from tqdm import tqdm

class Fingerprint:
    def __init__(self,params):
        self.params = params
        pass


    def smile_to_mol(self, smile:str) -> Optional[Mol]:
        """
        Converts a list of SMILES strings to RDKit Mol objects.
        :param smiles: List of SMILES strings.
        :return: List of RDKit Mol objects.
        """
        
        try:
            mol = MolFromSmiles(smile)
        except:
            mol = None
        return mol
    

    def create_custom_factory(self) -> SigFactory:
        factory = SigFactory.SigFactory()

        if self.params.xml is not None:
            if isinstance(self.params.xml, str):
                factory.LoadFromFile(self.params.xml)
            elif isinstance(self.params.xml, list):
                for xml_file in self.params.xml:
                    factory.LoadFromFile(xml_file)
        
        if self.params.donors:
            if isinstance(self.params.donors, str):
                factory.AddFeature('Donor', self.params.donors)  # Add donor features
            elif isinstance(self.params.donors, list):
                for donor in self.params.donors:
                    factory.AddFeature('Donor', donor)  # Add donor features
        if self.params.acceptors:
            if isinstance(self.params.acceptors, str):
                factory.AddFeature('Acceptor', self.params.acceptors)  # Add acceptor features
            elif isinstance(self.params.acceptors, list):
                for acceptor in self.params.acceptors:
                    factory.AddFeature('Acceptor', acceptor)  # Add acceptor features
        if self.params.aromatics:
            if isinstance(self.params.aromatics, str):
                factory.AddFeature('Aromatic', self.params.aromatics)  # Add acceptor features
            elif isinstance(self.params.aromatics, list):
                for aromatic in self.params.aromatics:
                    factory.AddFeature('Aromatic', aromatic)  # Add acceptor features
        if self.params.aliphatics:
            if isinstance(self.params.aliphatics, str):
                factory.AddFeature('Aliphatic', self.params.aliphatics)  # Add acceptor features
            elif isinstance(self.params.aliphatics, list):
                for aliphatic in self.params.aliphatics:
                    factory.AddFeature('Aliphatic', aliphatic)  # Add acceptor features
        if self.params.ringatoms:
            if isinstance(self.params.ringatoms, str):
                factory.AddFeature('RingAtom', self.params.ringatoms)  # Add ring features
            elif isinstance(self.params.ringatoms, list):
                for ringbond in self.params.ringatoms:
                    factory.AddFeature('RingAtom', ringbond)  # Add ring features
        if self.params.ringbonds:
            if isinstance(self.params.ringbonds, str):
                factory.AddFeature('RingBond', self.params.ringbonds)  # Add ring features
            elif isinstance(self.params.ringbonds, list):
                for ringbond in self.params.ringbonds:
                    factory.AddFeature('RingBond', ringbond)  # Add ring features
        if self.params.atoms:
            if isinstance(self.params.atoms, str):
                factory.AddFeature('Atoms', self.params.atoms)  # Add self.params.atoms features
            elif isinstance(self.params.atoms, list):
                for atom in self.params.atoms:
                    factory.AddFeature('Atoms', atom)  # Add atoms features
        if self.params.bonds:
            if isinstance(self.params.bonds, str):
                factory.AddFeature('Bonds', self.params.bonds)  # Add self.params.bonds features
            elif isinstance(self.params.bonds, list):
                for bond in self.params.bonds:
                    factory.AddFeature('Bonds', bond)  # Add bonds features
        
        if self.params.bins:
            factory.SetBins(self.params.bins)
        
    
        # Initialize the factory with these features
        factory.Init()

        return factory
    
    def smiles_to_fps(self, smiles: List[str],vars:dict) -> np.array:
        """
        Converts a list of SMILES strings to their corresponding EState fingerprints.
        :param smiles: List of SMILES strings.
        :return: List of EState fingerprints as UIntSparseIntVect.
        """
        fps = []
        for smile in tqdm(smiles, total=len(smiles),desc="Processing SMILES"):
            mol = self.smile_to_mol(smile)
            if mol is None:
                fps.append(None)
            else:
                fps.append(self.fpfunc(
                    mol,
                    fpname=vars['fpname'],
                    variant=vars['variant'],
                    radius=vars['radius'],
                    use_counts=vars['use_counts'],
                    use_features=vars['use_features'],
                    size=vars['size'],
                    fuzziness=vars['fuzziness'],
                    paths=vars['paths'],
                    author=vars['author'],
                    model=vars['model'],
                    padding=vars['padding']
                ))
        return fps
    
    def uint_to_array(self, uint: UIntSparseIntVect) -> np.array:
        """
        Converts a UIntSparseIntVect to a numpy array.
        :param uint: UIntSparseIntVect object.
        :return: Numpy array.
        """
        arr = np.zeros(uint.GetLength(), dtype=int)
        for idx, value in uint.GetNonzeroElements().items():
            arr[idx] = value
        return arr
    
    def bitvect_to_array(self, bit_vect: DataStructs.cDataStructs.ExplicitBitVect) -> np.array:
        arr = np.zeros((bit_vect.GetNumBits(),), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(bit_vect, arr)
        return arr

    def fpfunc(self,mol,fpname:str='ecfp',variant:str = 'raw_bits',radius:int=3,use_counts:bool=True,use_features:bool=True,size:int=2048,fuzziness:float = .5,paths:int=5,author:str = '',model:str = '',padding:bool=True):
        if fpname == 'ecfp':
            fp = AllChem.GetMorganFingerprint(mol, radius, useCounts=use_counts, useFeatures=use_features)
            fp = self.uint_to_array(fp)
            return fp
        elif fpname == 'hashed-ecfp':
            fp = AllChem.GetHashedMorganFingerprint(mol, radius, useCounts=use_counts, useFeatures=use_features)
            return fp
        elif fpname == 'bcut':
            return BCUT2D(mol)
        if fpname == 'rdkit':
            fp = RDKFingerprint(mol, maxPath=int(variant))
            return self.bitvect_to_array(fp)
        elif fpname == 'maccs':
            return self.bitvect_to_array(MACCSkeys.GenMACCSKeys(mol))
        elif fpname == 'pubchem':
            return self.uint_to_array(GetAtomPairFingerprint(mol))
        elif fpname == 'layered':
            return self.bitvect_to_array(LayeredFingerprint(mol))
        elif fpname == 'estate':
            return FingerprintMol(mol)[0]
        elif fpname == 'torsional':
            return self.uint_to_array(GetTopologicalTorsionFingerprintAsIntVect(mol))
        elif fpname == 'avalon':
            return self.bitvect_to_array(GetAvalonFP(mol))
        elif fpname == '2dpharm':
            factory = Gobbi_Pharm2D.factory
            return self.uint_to_array(Generate.Gen2DFingerprint(mol, factory))
        elif fpname == 'mapchiral':
            return encode(mol, max_radius=radius, n_permutations=size, mapping=False)
        elif fpname == 'mxfp':
            calculator = MXFPCalculator()
            return calculator.mxfp_from_mol(mol)
        elif fpname == 'autocorr':
            fp = AutocorrFingerprint()
            smiles = Chem.MolToSmiles(mol)
            return fp.transform([smiles])
        elif fpname == 'erg':
            fp = ERGFingerprint(fuzz_increment=fuzziness,max_path=paths)
            smiles = Chem.MolToSmiles(mol)
            return fp.transform([smiles])
        elif fpname == 'crippen':
            fp = GhoseCrippenFingerprint(count=use_counts)
            smiles = Chem.MolToSmiles(mol)
            return fp.transform([smiles])
        elif fpname == 'roth':
            fp = KlekotaRothFingerprint(count=use_counts)
            smiles = Chem.MolToSmiles(mol)
            return fp.transform([smiles])
        elif fpname == 'laggner':
            fp = LaggnerFingerprint(count=use_counts)
            smiles = Chem.MolToSmiles(mol)
            return fp.transform([smiles])
        elif fpname == 'lingo':
            fp = LingoFingerprint(substring_length=int(variant), count=use_counts)
            smiles = Chem.MolToSmiles(mol)
            return fp.transform([smiles])
        elif fpname == 'map4':
            fp = MAPFingerprint(radius=radius, variant=variant)
            smiles = Chem.MolToSmiles(mol)
            return fp.transform([smiles])
        elif fpname == 'mhfp':
            fp = MHFPFingerprint(radius=radius, min_radius=int(variant), variant=variant)
            smiles = Chem.MolToSmiles(mol)
            return fp.transform([smiles])
        elif fpname == 'morse':
            fp = MORSEFingerprint()
            smiles = Chem.MolToSmiles(mol)
            return fp.transform([smiles])
        elif fpname == 'mordred':
            fp = MordredFingerprint()
            smiles = Chem.MolToSmiles(mol)
            return fp.transform([smiles])
        elif fpname == 'mqns':
            fp = MQNsFingerprint(count=use_counts)
            smiles = Chem.MolToSmiles(mol)
            return fp.transform([smiles])
        elif fpname == 'patternfp':
            fp = PatternFingerprint()
            smiles = Chem.MolToSmiles(mol)
            return fp.transform([smiles])
        elif fpname == 'physchemfp':
            fp = PhysiochemicalPropertiesFingerprint(variant=variant, count=use_counts)
            smiles = Chem.MolToSmiles(mol)
            return fp.transform([smiles])
        elif fpname == 'secfp':
            fp = SECFPFingerprint(radius=radius, min_radius=int(variant))
            smiles = Chem.MolToSmiles(mol)
            return fp.transform([smiles])
        elif fpname == 'vsa':
            fp = VSAFingerprint(variant=variant)
            smiles = Chem.MolToSmiles(mol)
            return fp.transform([smiles])
        elif fpname == 'whim':
            fp = WHIMFingerprint()
            smiles = Chem.MolToSmiles(mol)
            return fp.transform([smiles])
    
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
                return tokenizer(smiles,padding=padding,return_tensors='pt')
            else:
                tokenizer = AutoTokenizer.from_pretrained(checkpoint)
                if 'selfies' in model:
                    smiles = self.smiles_to_selfies(smiles)
                return tokenizer(smiles,return_tensors='pt')
    
    def PyFingerprintUserFunction(self,smiles:str,function:str) -> List[UIntSparseIntVect]:
        print(smiles)
        try:
            fp = get_fingerprints([smiles],function)
            return np.array(fp)
        except:
            return None
    
    
    def smiles_to_selfies(self, smiles: List[str]) -> List[str]:
        """
        Converts a list of SMILES strings to their corresponding SELFIES strings.
        :param smiles: List of SMILES strings.
        :return: List of SELFIES strings.
        """
        selfies_list = [selfies_encoder(smile) for smile in smiles]
        return selfies_list