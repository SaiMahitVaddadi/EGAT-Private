
from .molmatdesc import MolMatDesc
from rdkit import Chem
from .properties import Properties

class StereoChemistry:
    def __init__(self,egatecule:MolMatDesc):
        self.new_mol = egatecule.new_mol
        self.mol_sanitized = egatecule.mol_sanitized

        # initialize 
        self.bond_stereo    = {}
        self.chiral_centers = {}
        self.Hybridization  = {}
        self.conjugation    = {}

    def ObtainChiralCenters(self):
        # obtain chiral centers
        for chiral_center in self.chiral_centers_sanitized:
            atom_index, chirality = chiral_center
            atom_map_num = self.mol_sanitized.GetAtomWithIdx(atom_index).GetAtomMapNum()
            self.chiral_centers[atom_map_num] = chirality

    def ChiralCenters(self):
        if self.mol_sanitized is not None:
            self.chiral_centers_sanitized = Chem.FindMolChiralCenters(self.mol_sanitized, includeUnassigned=True)
        else:
            self.chiral_centers_sanitized = Chem.FindMolChiralCenters(self.new_mol, includeUnassigned=True)
        self.ObtainChiralCenters()

    def BondIndex(self,bond):
        begin_atom = bond.GetBeginAtom().GetAtomMapNum()
        end_atom = bond.GetEndAtom().GetAtomMapNum()
        bond_index = tuple(sorted([begin_atom,end_atom]))
        return begin_atom,end_atom,bond_index
    
    def EncodeBondStereo(self,bond):
        begin_atom,end_atom,bond_index = self.BondIndex(bond)
        self.conjugation[bond_index] = bond.GetIsConjugated()
        self.bond_aromatic[bond_index] = bond.GetIsAromatic()

        stereo = bond.GetStereo()
        if stereo != Chem.BondStereo.STEREONONE:
            self.bond_stereo_info[(begin_atom, end_atom)] = stereo

    def BondStereo(self):
        self.bond_stereo_info = {}
        if self.mol_sanitized is not None:
            # Get the bond stereo information from the sanitized molecule
            for bond in self.mol_sanitized.GetBonds(): self.EncodeBondStereo(bond)
        else:
            for bond in self.new_mol.GetBonds(): self.EncodeBondStereo(bond)
        self.BondStereoComparison()

    
    def BondStereoComparison(self):
         # Compare with the original molecule
        for bond in self.new_mol.GetBonds():
            begin_atom,end_atom,bond_index = self.BondIndex(bond)

            if (begin_atom, end_atom) in self.bond_stereo_info:
                stereo = self.bond_stereo_info[(begin_atom, end_atom)]

                if stereo == Chem.BondStereo.STEREOE:
                    self.bond_stereo[bond_index] = 'E'
                elif stereo == Chem.BondStereo.STEREOZ:
                    self.bond_stereo[bond_index] = 'Z'
                elif stereo == Chem.BondStereo.STEREOANY:
                    self.bond_stereo[bond_index] = 'ANY'

    def EncodeAtomStereo(self,atom):
        atom_map_num = atom.GetAtomMapNum()
        self.atom_aromatic[atom_map_num] = atom.GetIsAromatic()
        self.Hybridization[atom_map_num] = atom.GetHybridization()
        
    def AtomStereo(self):
        # go through heavy atoms
        if self.mol_sanitized is not None:
            for atom in self.mol_sanitized.GetAtoms(): self.EncodeAtomStereo(atom)
        else:
            for bond in self.new_mol.GetBonds(): self.EncodeBondStereo(bond)

    def run(self):
        self.ChiralCenters()
        self.BondStereo()
        self.AtomStereo()
        