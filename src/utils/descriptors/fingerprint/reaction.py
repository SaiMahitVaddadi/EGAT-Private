import random
from typing import List, Tuple, Optional

from drfp import DrfpEncoder
from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator, get_default_model_and_tokenizer, generate_fingerprints
)
from rdkit import Chem



class ReactionFingerprint:
    def __init__(self):
        self.model, self.tokenizer = get_default_model_and_tokenizer()
        self.rxnfp_generator = RXNBERTFingerprintGenerator(self.model, self.tokenizer)
        pass
    
    def smiles_to_fps(self, smiles: List[str], fpname: str = 'ecfp', n_folded_length: int = 2048, min_radius: int = 0, radius: int = 3, rings: bool = True, mapping: bool = False, atom_index_mapping: bool = False, root_central_atom: bool = True, include_hydrogens: bool = False, show_progress_bar: bool = False) -> List:
        """
        Converts a list of SMILES strings to their corresponding fingerprints.
        :param smiles: List of SMILES strings.
        :param fpname: Fingerprint name.
        :return: List of fingerprints.
        """
        fps = []
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fps.append(self.fpfunc(mol, fpname=fpname, n_folded_length=n_folded_length, min_radius=min_radius, radius=radius, rings=rings, mapping=mapping, atom_index_mapping=atom_index_mapping, root_central_atom=root_central_atom, include_hydrogens=include_hydrogens, show_progress_bar=show_progress_bar))
        return fps


    def fpfunc(self, mol, fpname: str = 'ecfp', n_folded_length: int = 2048, min_radius: int = 0, radius: int = 3, rings: bool = True, mapping: bool = False, atom_index_mapping: bool = False, root_central_atom: bool = True, include_hydrogens: bool = False, show_progress_bar: bool = False):
        if fpname == 'drfp':
            fps = DrfpEncoder.encode(
                mol,
                n_folded_length=n_folded_length,
                min_radius=min_radius,
                radius=radius,
                rings=rings,
                mapping=mapping,
                atom_index_mapping=atom_index_mapping,
                root_central_atom=root_central_atom,
                include_hydrogens=include_hydrogens,
                show_progress_bar=show_progress_bar
            )
        elif fpname == 'rxnfp':
            fps = self.rxnfp_generator.convert(mol)
        return fps
    
