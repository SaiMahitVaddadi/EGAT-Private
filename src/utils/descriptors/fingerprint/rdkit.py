"""Compute various scores with RDKit"""

from typing import List, Callable

import numpy as np
from rdkit.Chem import AllChem as Chem, Crippen, Descriptors, Lipinski

def num_sp(mol: Chem.Mol) -> int:
    num_sp_atoms = len(
        [atom for atom in mol.GetAtoms() if atom.GetHybridization() == Chem.HybridizationType.SP]
    )

    return num_sp_atoms

def num_sp2(mol: Chem.Mol) -> int:
    num_sp2_atoms = len(
        [atom for atom in mol.GetAtoms() if atom.GetHybridization() == Chem.HybridizationType.SP2]
    )

    return num_sp2_atoms

def num_sp3(mol: Chem.Mol) -> int:
    num_sp3_atoms = len(
        [atom for atom in mol.GetAtoms() if atom.GetHybridization() == Chem.HybridizationType.SP3]
    )
    return num_sp3_atoms

def graph_length(mol: Chem.Mol) -> int:
    return int(np.max(Chem.GetDistanceMatrix(mol)))

cls_func_map = {
    "Qed": Descriptors.qed,
    "MolecularWeight": Descriptors.MolWt,
    "GraphLength": graph_length,
    "NumAtomStereoCenters": Chem.CalcNumAtomStereoCenters,
    "HBondAcceptors": Lipinski.NumHAcceptors,
    "HBondDonors": Lipinski.NumHDonors,
    "NumRotBond": Lipinski.NumRotatableBonds,
    "Csp3": Lipinski.FractionCSP3,
    "numsp": num_sp,
    "numsp2": num_sp2,
    "numsp3": num_sp3,
    "NumHeavyAtoms": Lipinski.HeavyAtomCount,
    "NumHeteroAtoms": Lipinski.NumHeteroatoms,
    "NumRings": Lipinski.RingCount,
    "NumAromaticRings": Lipinski.NumAromaticRings,
    "NumAliphaticRings": Lipinski.NumAliphaticRings,
    "SlogP": Crippen.MolLogP,
}
