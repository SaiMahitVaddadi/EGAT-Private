
from .molmatdesc import MolMatDesc
from rdkit import Chem
from .properties import Properties
import numpy as np
class StereoChemistry:
    def __init__(self,egatecule:MolMatDesc,v2=False):
        self.new_mol = egatecule.new_mol
        self.mol_sanitized = egatecule.mol_sanitized

        # initialize 
        self.bond_stereo    = {}
        self.chiral_centers = {}
        self.Hybridization  = {}
        self.conjugation    = {}
        self.bond_aromatic  = {}
        self.v2 = v2

    def ObtainChiralCenters(self):
        # obtain chiral centers
        for chiral_center in self.chiral_centers_sanitized:
            atom_index, chirality = chiral_center
            atom_map_num = self.mol_sanitized.GetAtomWithIdx(atom_index).GetAtomMapNum()
            self.chiral_centers[atom_map_num] = chirality

    def ChiralCenters(self):
        if isinstance(self.mol_sanitized,Chem.rdchem.Mol):
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

        if self.v2: func = self.EncodeBondStereov2
        else: func = self.EncodeBondStereo
        if isinstance(self.mol_sanitized,Chem.rdchem.Mol):
            # Get the bond stereo information from the sanitized molecule
            for bond in self.mol_sanitized.GetBonds(): func(bond)
        else:
            for bond in self.new_mol.GetBonds(): func(bond)
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

    def EncodeBondStereov2(self,bond):
        begin_atom,end_atom,bond_index = self.BondIndex(bond)
        self.conjugation[bond_index] = bond.GetIsConjugated()
        self.bond_aromatic[bond_index] = bond.GetIsAromatic()

        stereo = bond.GetStereo()
        self.bond_stereo_info[(begin_atom, end_atom)] = stereo

    def BondStereoComparisonNew(self):
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
                elif stereo == Chem.BondStereo.STEREOATROPCCW:
                    self.bond_stereo[bond_index] = 'ATROPCCW'
                elif stereo == Chem.BondStereo.STEREOATROPCW:
                    self.bond_stereo[bond_index] = 'ATROPCW'
                elif stereo == Chem.BondStereo.STEREOATRANS:
                    self.bond_stereo[bond_index] = 'TRANS'
                elif stereo == Chem.BondStereo.STEREOCIS:
                    self.bond_stereo[bond_index] = 'CIS'
                elif stereo == Chem.BondStereo.STEREONONE:
                    self.bond_stereo[bond_index] = 'NONE'
    


    def AtomChirality(self):
        self.atom_chiral = {}
        for atom in self.new_mol.GetAtoms():
            atom_map_num = atom.GetAtomMapNum()
            chiral_tag = atom.GetChiralTag()
            try:
                self.atom_chiral[atom_map_num] = self.atom_chiral_encode[chiral_tag]
            except:
                self.atom_chiral[atom_map_num] = self.atom_chiral_encode[Chem.rdchem.ChiralType.CHI_UNSPECIFIED]

    def AtomStereoType(self):
        self.atom_stereo = {}
        self.bond_stereo_v2 = {}
        self.atom_stereo_bond = {}
        self.bond_stereo_bond = {}
        self.global_bond_stereo = {}

        for e in Chem.rdmolops.FindPotentialStereo(self.new_mol):
            stereo_description = {e.centeredOn:e.descriptor}
            stereo_description_bond = {e.centeredOn:e.type}
            stereo_atoms_from_cip = {e.centeredOn:[c for c in e.controllingAtoms if c < 1000]}

            atom_index = e.centeredOn
            atom_map_num = self.new_mol.GetAtomWithIdx(atom_index).GetAtomMapNum()
            
            
            if atom_map_num not in self.atom_stereo:
                self.atom_stereo[atom_map_num] = self.atom_chiral[stereo_description[atom_index]]
                self.atom_stereo_bond[atom_map_num] = stereo_description_bond[atom_index]
            else:
                self.atom_stereo[atom_map_num] = list(np.array(self.atom_stereo[atom_map_num]) + np.array(stereo_description[atom_index]))
                self.atom_stereo_bond[atom_map_num] = list(np.array(self.atom_stereo_bond[atom_map_num]) + np.array(stereo_description_bond[atom_index]))
            for c in e.controllingAtoms:
                if c < 1000 and self.new_mol.GetBondBetweenAtoms(atom_index, c) is not None:
                    c = self.new_mol.GetAtomWithIdx(atom_index).GetAtomMapNum()
                    if (atom_map_num, c) not in self.bond_stereo_v2:
                        self.bond_stereo_v2[(atom_map_num, c)] = self.atom_chiral[stereo_description[atom_index]]
                        self.bond_stereo_bond[(atom_map_num, c)] = stereo_description_bond[atom_index]
                    else:
                        self.bond_stereo_v2[(atom_map_num, c)] = list(np.array(self.atom_stereo[atom_map_num]) + np.array(stereo_description[atom_index]))
                        self.bond_stereo_bond[(atom_map_num, c)] = list(np.array(self.bond_stereo_bond[(atom_map_num, c)]) + np.array(stereo_description_bond[atom_index]))
                
            self.global_bond_stereo[(stereo_atoms_from_cip[0],stereo_atoms_from_cip[2])] = self.atom_chiral[stereo_description[atom_index]]


    def EncodeAtomStereo(self,atom):
        atom_map_num = atom.GetAtomMapNum()
        self.atom_aromatic[atom_map_num] = atom.GetIsAromatic()
        self.Hybridization[atom_map_num] = atom.GetHybridization()
        
    def AtomStereo(self):
        # go through heavy atoms
        if isinstance(self.mol_sanitized,Chem.rdchem.Mol):
            for atom in self.mol_sanitized.GetAtoms(): self.EncodeAtomStereo(atom)
        else:
            for bond in self.new_mol.GetBonds(): self.EncodeBondStereo(bond)

    def run(self):
        self.ChiralCenters()
        self.BondStereo()
        self.AtomStereo()
        self.AtomStereoType()