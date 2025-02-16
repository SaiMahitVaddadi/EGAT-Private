"""Compute various scores with RDKit"""
import numpy as np
from rdkit.Chem import AllChem as Chem, Crippen, Descriptors, Lipinski





class RDKitDescriptors:
    def __init__(self, calc='rdkit',sandp=False,grid_spacing = .2,box_margin=.2):
        self.calc = calc
        self.descriptors = {
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
    "SlogP": Crippen.MolLogP}
        self.sandp = sandp
        self.grid_spacing = grid_spacing
        self.box_margin = box_margin

    def _computefunc(self, mol: Chem.Mol, descriptor):
        if descriptor == 'tpsa':
            return self.tpsa(mol)
        elif descriptor == 'volume':
            return self.volume(mol)
        else:
            return self.descriptors[descriptor](mol)

    def compute(self, mol: Chem.Mol,descriptor) -> list:
        if isinstance(descriptor, str):
            return self._computefunc(mol, descriptor)
        elif isinstance(descriptor, list):
            return [self._computefunc(mol, descriptor) for desc in descriptor]
        
    def tpsa(self, mol: Chem.Mol) -> float:
        return Descriptors.TPSA(mol, includeSandP=self.sandp)
    
    def volume(self, mol: Chem.Mol,) -> float:
        mol3d = Chem.AddHs(mol)  # will not consider protonation state
        Chem.EmbedMolecule(mol3d)
        volume = Chem.ComputeMolVolume(
            mol3d, gridSpacing=self.grid_spacing, boxMargin=self.box_margin
        )
        return volume
        

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

