from rdkit import Chem 
import numpy as np
from rdkit.Chem import rdmolops
from rdkit.Chem import AllChem

"""
MolMatDesc
==========

A class to generate molecular descriptors from atom-mapped SMILES strings.

Methods
-------

__init__(AM_smiles)
    Initialize the MolMatDesc object with a SMILES string.

AddAtoms()
    Add sorted atoms to the new molecule and create an atom map.

AddBonds()
    Add bonds to the new molecule using the atom map.

CleanFinalMolecule()
    Finalize the molecule and remove atom mapping numbers.

BondMatrix()
    Create adjacency and bond order matrices for the molecule.

Elements()
    Get the formal charges and element symbols of the atoms.

GetMatrix()
    Generate the molecule matrix and sanitize the molecule.

ReturnAMSmiles()
    Convert the molecule to SMILES format.

RotatableBondCount()
    Count the number of rotatable bonds in the molecule.

readenegtable()
    Read the electronegativity table from a file.

Electronegativity()
    Get the electronegativity values of the atoms.

Spiro()
    Find the location of spiro atoms in the molecule.

BridgeHead()
    Find the location of bridgehead atoms in the molecule.

BondPolarityPauling()
    Calculate the bond polarity using Pauling electronegativity values.

GenerateRandom3DGeometry()
    Generate a random 3D geometry for the molecule.

Gasteiger()
    Compute Gasteiger charges for the molecule.

BondDipoleMoments()
    Calculate the dipole moments for each bond in the molecule.

run()
    Execute the main workflow of the MolMatDesc class.


Attributes
----------
    mol: RDKit molecule object.
    sorted_atoms: List of sorted atom objects.
    new_mol: RDKit editable molecule object.
    smiles: Atom-mapped SMILES string.
    atom_map: Dictionary of atom mapping numbers.
    adj_mat: Numpy array of adjacency matrix.
    bond_mat: Numpy array of bond order matrix.
    fc: List of formal charges.
    element: List of element symbols.
    mol_sanitized: RDKit molecule object.
    num_atoms: Number of atoms in the molecule.
    electronegativity: List of electronegativity values.
    spiro: List of spiro atom indices.
    bridgehead: List of bridgehead atom indices.
    polarity_pauling: List of bond polarity values.
    bond_dipole_moments: List of bond dipole moments.
    rotationalbond: List of rotatable bond indices.
    new_smiles: SMILES string of the new molecule.
    sanitized_smiles: SMILES string of the sanitized molecule.
    

"""


class MolMatDesc:
    def __init__(self, AM_smiles):
        """
        Initialize the MolMatDesc object with a SMILES string.
        
        Parameters:
        AM_smiles (str): Atom-mapped SMILES string of the molecule.
        """
        self.mol = Chem.MolFromSmiles(AM_smiles, sanitize=False)
        self.AddMapping()
        self.sorted_atoms = sorted(self.mol.GetAtoms(), key=lambda atom: atom.GetAtomMapNum())
        self.new_mol = Chem.EditableMol(Chem.Mol())
        self.smiles = AM_smiles 

    def AddMapping(self):
        for i, atom in enumerate(self.mol.GetAtoms()):
            if atom.GetAtomMapNum() == 0:
                atom.SetAtomMapNum(i + 1)

    def AddAtoms(self):
        """
        Add sorted atoms to the new molecule and create an atom map.
        """
        self.atom_map = {}
        for atom in self.sorted_atoms:
            idx = self.new_mol.AddAtom(atom)
            self.atom_map[atom.GetAtomMapNum()] = idx

    def AddBonds(self):
        """
        Add bonds to the new molecule using the atom map.
        """
        for bond in self.mol.GetBonds():
            begin_atom = self.atom_map[bond.GetBeginAtom().GetAtomMapNum()]
            end_atom = self.atom_map[bond.GetEndAtom().GetAtomMapNum()]
            bond_type = bond.GetBondType()
            self.new_mol.AddBond(begin_atom, end_atom, bond_type)
    
    def CleanFinalMolecule(self):
        """
        Finalize the molecule and remove atom mapping numbers.
        """
        self.new_mol = self.new_mol.GetMol()
        for atom in self.new_mol.GetAtoms():
            atom.SetAtomMapNum(0)
        self.num_atoms = self.new_mol.GetNumAtoms()

    def BondMatrix(self):
        """
        Create adjacency and bond order matrices for the molecule.
        """
        self.adj_mat = np.zeros((self.num_atoms, self.num_atoms), dtype=int)
        self.bond_mat = np.zeros((self.num_atoms, self.num_atoms), dtype=int)
        for bond in self.new_mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            self.adj_mat[i, j] = 1
            self.adj_mat[j, i] = 1
            bond_order = int(bond.GetBondTypeAsDouble())
            self.bond_mat[i, j] = bond_order
            self.bond_mat[j, i] = bond_order

    def Elements(self):
        """
        Get the formal charges and element symbols of the atoms.
        """
        self.fc = []
        self.element = []
        for atom in self.new_mol.GetAtoms():
            formal_charge = atom.GetFormalCharge()
            self.fc.append(formal_charge)
            self.element.append(atom.GetSymbol())

    def GetMatrix(self):
        """
        Generate the molecule matrix and sanitize the molecule.
        """
        self.AddAtoms()
        self.AddBonds()
        self.CleanFinalMolecule()
        self.BondMatrix()
        self.Elements()
        try:
            self.mol_sanitized = Chem.SanitizeMol(self.new_mol)
        except:
            self.mol_sanitized = Chem.MolFromSmiles(self.smiles)

    def ReturnAMSmiles(self):
        """
        Convert the molecule to SMILES format.
        """
        self.new_smiles = Chem.MolToSmiles(self.new_mol, isomericSmiles=True, kekuleSmiles=False, allBondsExplicit=False, allHsExplicit=False)
        try:
            self.sanitized_smiles = Chem.MolToSmiles(self.mol_sanitized, isomericSmiles=True, kekuleSmiles=False, allBondsExplicit=False, allHsExplicit=False)
        except:
            self.sanitized_smiles = self.new_smiles
    def RotatableBondCount(self):
        """
        Count the number of rotatable bonds in the molecule.
        """
        RotatableBond = Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')
        self.rotationalbond = self.new_mol.GetSubstructMatches(RotatableBond)

    def readenegtable(self):
        """
        Read the electronegativity table from a file.
        
        Returns:
        dict: Dictionary of electronegativity values.
        list: List of element symbols in the dictionary.
        """
        with open('/Users/svaddadi/Documents/GitHub/EGAT/src/utils/database/pauling.txt', 'r') as file:
            lines = file.readlines()
        electronegativity_dict = {}
        for line in lines:
            parts = [part.strip() for part in line.split('#') if part.strip()]
            if len(parts) == 2:
                electronegativity = 0 if parts[0] == '-' else float(parts[0])
                element = parts[1].split(' ')[1]
                electronegativity_dict[element] = electronegativity
        symbolsindict = list(electronegativity_dict.keys())
        return electronegativity_dict, symbolsindict

    def Electronegativity(self):
        """
        Get the electronegativity values of the atoms.
        """
        electronegativity_dict, symbolsindict = self.readenegtable()
        atoms = self.new_mol.GetAtoms()
        electronegativity_list = []
        for atom in atoms:
            symbol = atom.GetSymbol()
            if symbol in symbolsindict:
                electronegativity_list.append(electronegativity_dict[symbol])
            else:
                electronegativity_list.append(0)
        self.electronegativity = electronegativity_list

    def InitializeUniqueBonds(self):
        if not self.new_mol.GetRingInfo() or self.new_mol.GetRingInfo().NumRings() == 0:
            rdmolops.FindSSSR(self.new_mol)
        rInfo = self.new_mol.GetRingInfo()
        atoms = []
        lAtoms = []
        if not atoms:
            atoms = lAtoms
        return rInfo, atoms, lAtoms

    
    def Spiro(self):
        """
        Find the location of spiro atoms in the molecule.
        """
        rInfo, atoms, lAtoms = self.InitializeUniqueBonds()
        for i in range(len(rInfo.AtomRings())):
            ri = rInfo.AtomRings()[i]
            for j in range(i + 1, len(rInfo.AtomRings())):
                rj = rInfo.AtomRings()[j]
                inter = list(set(ri).intersection(rj))
                if len(inter) == 1:
                    if inter[0] not in atoms:
                        atoms.append(inter[0])
        self.spiro = atoms 

    def BridgeHead(self):
        """
        Find the location of bridgehead atoms in the molecule.
        """
        rInfo, atoms, lAtoms = self.InitializeUniqueBonds()
        for i in range(len(rInfo.BondRings())):
            ri = rInfo.BondRings()[i]
            for j in range(i + 1, len(rInfo.BondRings())):
                rj = rInfo.BondRings()[j]
                inter = set(ri).intersection(rj)
                if len(inter) > 1:
                    atomCounts = [0] * self.new_mol.GetNumAtoms()
                    for ii in inter:
                        atomCounts[self.new_mol.GetBondWithIdx(ii).GetBeginAtomIdx()] += 1
                        atomCounts[self.new_mol.GetBondWithIdx(ii).GetEndAtomIdx()] += 1
                    for ti in range(len(atomCounts)):
                        if atomCounts[ti] == 1:
                            if ti not in atoms:
                                atoms.append(ti)
        self.bridgehead = atoms 
    
    def BondPolarityPauling(self):
        """
        Calculate the bond polarity using Pauling electronegativity values.
        """
        electronegativity_dict, symbolsindict = self.readenegtable()
        bonds = self.new_mol.GetBonds()
        polarity_info = []
        for bond in bonds:
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            begin_symbol = begin_atom.GetSymbol()
            end_symbol = end_atom.GetSymbol()
            if begin_symbol in symbolsindict and end_symbol in symbolsindict:
                begin_en = electronegativity_dict[begin_symbol]
                end_en = electronegativity_dict[end_symbol]
                en_difference = abs(begin_en - end_en)
                polarity_info.append(((begin_symbol, end_symbol), en_difference))
            else:
                polarity_info.append(((begin_symbol, end_symbol), 0))
        self.polarity_pauling = polarity_info
    
    def GenerateRandom3DGeometry(self):
        """
        Generate a random 3D geometry for the molecule.
        """
        Chem.AllChem.EmbedMolecule(self.new_mol, randomSeed=42)
        Chem.AllChem.MMFFOptimizeMolecule(self.new_mol)
        
    def Gasteiger(self):
        """
        Compute Gasteiger charges for the molecule.
        """
        self.GenerateRandom3DGeometry()
        Chem.rdPartialCharges.ComputeGasteigerCharges(self.new_mol)
        self.gasteiger_charges = [atom.GetProp('_GasteigerCharge') for atom in self.new_mol.GetAtoms()]

    def BondDipoleMoments(self):
        """
        Calculate the dipole moments for each bond in the molecule.
        """
        if not hasattr(self, 'gasteiger_charges'):
            self.Gasteiger()
        conf = self.new_mol.GetConformer()
        coords = np.array([list(conf.GetAtomPosition(i)) for i in range(self.num_atoms)])
        bond_dipole_moments = []
        for bond in self.new_mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            charge_diff = float(self.gasteiger_charges[begin_idx]) - float(self.gasteiger_charges[end_idx])
            bond_vector = coords[end_idx] - coords[begin_idx]
            bond_dipole = charge_diff * bond_vector
            bond_dipole_moments.append(np.linalg.norm(bond_dipole))
        self.bond_dipole_moments = bond_dipole_moments

    def run(self):
        """
        Execute the main workflow of the MolMatDesc class.
        """
        self.GetMatrix()
        self.ReturnAMSmiles()
        self.RotatableBondCount()
        self.Electronegativity()
        self.Spiro()
        self.BridgeHead()
        self.BondPolarityPauling()
        self.BondDipoleMoments()