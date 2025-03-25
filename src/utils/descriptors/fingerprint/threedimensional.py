import random
from typing import List, Tuple, Optional

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
from rdkit.Chem.rdmolops import RenumberAtoms
from rdkit.DataStructs.cDataStructs import UIntSparseIntVect
from molvs import Standardizer
from typing import List
from skfp.fingerprints import *
from skfp.preprocessing import MolFromSmilesTransformer,ConformerGenerator
import numpy as np
import e3fp 
from e3fp.pipeline import confs_from_smiles
from ..geometry.geometry import ConformerGeneratorEGAT
from mxfp.mxfp import MXFPCalculator
from ..egat.molmatdesc import MolMatDesc




class GeometricFingerprint:
    
    def ProcessFP(self, smiles: List[str], fp) -> np.array:
        """
        Process the fingerprints for the given SMILES strings.

        :param smiles: List of SMILES strings.
        :param fp: Fingerprint object to use for processing.
        :return: Array of processed fingerprints.
        """
        if 'mxfp' in fp.__class__.__name__.lower():
            fps = [] 
            for smi in smiles:
                descriptor = MolMatDesc(smi)
                descriptor.run()
                generator = ConformerGeneratorEGAT(descriptor)
                if descriptor.new_mol is not None:
                    fps.append(fp.mxfp_from_mol(descriptor.new_mol))
                else:
                    fps.append(None)
        else:
            mol_from_smiles = MolFromSmilesTransformer()
            mols = mol_from_smiles.transform(smiles)
            conf_gen = ConformerGenerator(n_jobs=-1)
            mols = conf_gen.transform(mols)
            fps = []
            for mol in mols:
                try:
                    # Transform to Mordred fingerprints
                    res = fp.transform([mol])[0]
                    res = np.where(np.isnan(res), -1000, res)
                    fps += [res]
                except Exception as e:
                    print(f"Error processing SMILES {mol}: {e}")
                    fps.append(None)
        fps = np.array(fps)
        return fps
    
    def smiles_to_fp(self, smiles: List[str], vars:dict) -> List[UIntSparseIntVect]:
        """
        Converts a list of SMILES strings to their corresponding fingerprints.

        :param smiles: List of SMILES strings.
        :param fpname: Name of the fingerprint to use.
        :param variant: Variant of the fingerprint, default is 'raw_bits'.
        :return: List of fingerprints as UIntSparseIntVect.
        """

        fpname = vars['fpname']
        variant = vars['variant']
        if fpname.upper() in ['E3FP', 'RDF', 'GETAWAY', 'USR', 'USRCAT', 'WHIM']:
            fp = getattr(f'{fpname.upper()}Fingerprint')()
        elif fpname == '3dpharm':
            fp = PharmacophoreFingerprint(variant=variant, use_3D=True)
        elif fpname == '3dmordred':
            fp = MordredFingerprint(use_3D=True)
        elif fpname == 'mxfp':
            fp = MXFPCalculator(dimensionality='3D')
        fps = self.ProcessFP(smiles, fp)
        return fps
