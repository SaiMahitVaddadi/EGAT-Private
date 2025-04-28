from .molmatdesc import MolMatDesc

from rdkit import Chem
from rdkit.Chem import rdDistGeom
import pandas as pd 
import sys 
from .....addons.yarp.yarp import yarpecule
from .properties import Properties
import numpy as np 

"""
:class:`RDKElectronInfo`
    :param egatecule: Molecule descriptor object
    :type egatecule: MolMatDesc

    .. method:: __init__(egatecule)
        :param egatecule: Molecule descriptor object
        :type egatecule: MolMatDesc
        :return: None

    .. method:: GetValence(atom)
        :param atom: Atom object
        :type atom: rdkit.Chem.rdchem.Atom
        :return: None

    .. method:: GetBonds(atom)
        :param atom: Atom object
        :type atom: rdkit.Chem.rdchem.Atom
        :return: Number of bonds
        :rtype: float

    .. method:: EncodeLonePair(atom)
        :param atom: Atom object
        :type atom: rdkit.Chem.rdchem.Atom
        :return: Atom map number, lone pairs, and radicals
        :rtype: tuple

    .. method:: ChangeForSpecialCases(lp, atom)
        :param lp: Lone pairs
        :type lp: float
        :param atom: Atom object
        :type atom: rdkit.Chem.rdchem.Atom
        :return: Adjusted lone pairs
        :rtype: float

    .. method:: GetElectronInfo()
        :return: None

    .. method:: run()
        :return: None
"""

class RDKElectronInfo:
    def __init__(self,egatecule:MolMatDesc):
        self.new_mol = egatecule.new_mol
        self.props = Properties()
        rdDistGeom.EmbedMolecule(self.new_mol,useRandomCoords=True,randomSeed=42)

    def GetValence(self,atom):
        symbol = atom.GetSymbol()
        symbol = symbol.lower()
        self.valence = self.props.el_valence[symbol]
        self.alt_valence = self.props.el_alt_valence[symbol]
    
    def GetBonds(self,atom):
        bonds = 0
        num_aromatic_bonds = 0
        for neighbor in atom.GetNeighbors():
            bond = self.new_mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            if bond.GetIsAromatic():
                bonds += 2.0
                num_aromatic_bonds += 1
            else:
                bonds += bond.GetBondTypeAsDouble()
        if bond.GetIsAromatic():
            if num_aromatic_bonds > 1:
                bonds = bonds - (num_aromatic_bonds - 1)
        return bonds
    
    def EncodeLonePair(self,atom):
        amap = atom.GetAtomMapNum()
        radicals = atom.GetNumRadicalElectrons()
        formal_charge = atom.GetFormalCharge()
        self.GetValence(atom)
        lp = self.valence - self.GetBonds(atom) - atom.GetNumRadicalElectrons() - atom.GetFormalCharge()
        lp = self.ChangeForSpecialCases(lp,atom)
        return amap,lp,radicals

    def ChangeForSpecialCases(self,lp,atom):
        if atom.GetSymbol() in ['P','S'] and lp < 0:
            lp = self.alt_valence - self.GetBonds(atom) - atom.GetNumRadicalElectrons() - atom.GetFormalCharge()
            if lp < 0: lp = 0
        else:
            if lp < 0 : lp = 0
        return lp

    def GetElectronInfo(self):
        amaplist = []
        lplist = []
        radlist = [] 
        for atom in self.new_mol.GetAtoms():
            amap,lp,radicals = self.EncodeLonePair(atom)
            amaplist.append(amap)
            lplist.append(lp)
            radlist.append(radicals)

        result = pd.DataFrame({'lp_atom_index':lplist,'mapping':amaplist,'radical_atom_index':radlist})
        result = result.sort_values('mapping')
        self.lps = result['lp_atom_index'].tolist()
        self.rads = result['radical_atom_index'].tolist()
        
        
    def run(self):
        self.GetElectronInfo()

'''
A class to handle electron information for a molecule using RDKit.
    Initializes the RDKElectronInfo object with a molecule descriptor.
    Retrieves the valence information for a given atom.
    Calculates the number of bonds for a given atom.
    Encodes lone pair information for a given atom.
    Adjusts lone pairs for special cases.
    Retrieves electron information for the molecule.
    Executes the electron information retrieval process.
:class:`YARPElectronInfo`
A class to handle electron information for a molecule using YARP.
:param canon: Canonicalization flag
:type canon: bool
:param mapping: Mapping flag
:type mapping: bool
.. method:: __init__(egatecule, canon=True, mapping=True)
    Initializes the YARPElectronInfo object with a molecule descriptor.
    :param canon: Canonicalization flag
    :type canon: bool
    :param mapping: Mapping flag
    :type mapping: bool
    Retrieves the valence information for a given atom.
.. method:: ValenceCheck(nbelectrons, bonds, fc, valence)
    Checks the valence for a given atom.
    :param nbelectrons: Number of electrons
    :type nbelectrons: int
    :param bonds: Number of bonds
    :type bonds: float
    :param fc: Formal charge
    :type fc: int
    :param valence: Valence
    :type valence: int
    :return: Lone pairs
.. method:: EncodeRadicalInfo(bm, i)
    Encodes radical information for a given atom.
    :param bm: Bond matrix
    :type bm: numpy.ndarray
    :param i: Atom index
    :type i: int
    :return: Lone pairs
.. method:: ChangeForSpecialCases(lp, atom, bm, row, i)
    Adjusts lone pairs for special cases.
    :param bm: Bond matrix
    :type bm: numpy.ndarray
    :param row: Bond matrix row
    :type row: numpy.ndarray
    :param i: Atom index
    :type i: int
.. method:: GetBonds(row, i)
    Calculates the number of bonds for a given atom.
    :param row: Bond matrix row
    :type row: numpy.ndarray
    :param i: Atom index
    :type i: int
    Retrieves electron information for the molecule.
.. method:: run()
    Executes the electron information retrieval process.
'''

class YARPElectronInfo: 
    def __init__(self,egatecule:MolMatDesc,canon:bool = True,mapping:bool = True):
        self.new_mol = egatecule.new_mol
        self.props = Properties()
        self.yarpecule = yarpecule(egatecule.new_smiles,canon,mapping) # this is a neat workaround, but we should see if we can just dump the molecule instead. 
        self.lps = []
        self.rads = []

    # check if we can pull atoms from the yarpecule
    def GetValence(self,atom):
        symbol = atom.GetSymbol()
        try:
            self.valence = self.props.el_valence[symbol.lower()]
        except:
            self.valence = self.props.el_valence[symbol]
        try:
            self.alt_valence = self.props.el_alt_valence[symbol.lower()]
        except:
            self.alt_valence = self.props.el_alt_valence[symbol]
    
    def ValenceCheck(self,nbelectrons,bonds,fc,valence):
        lp = valence - fc - bonds - nbelectrons
        return lp

    def EncodeRadicalInfo(self,bm,i):
        row = bm[i,:]
        atom = self.new_mol.GetAtoms()[i]
        self.GetValence(atom)
        lp = self.ValenceCheck(bm[i,i],self.GetBonds(row,i),self.yarpecule.fc[i],self.valence)
        lp = self.ChangeForSpecialCases(lp,atom,bm,row,i)
        return lp

    def ChangeForSpecialCases(self,lp,atom,bm,row,i):
        if atom.GetSymbol() in ['P','S'] and lp < 0:
            lp = self.ValenceCheck(bm[i,i],self.GetBonds(row,i),self.yarpecule.fc[i],self.alt_valence)
            if lp < 0: lp = bm[i,i]
        else:
            if lp < 0 : lp = bm[i,i]
        return lp
                
    def GetBonds(self,row,i):
        bonds = 0 
        for _,bond in enumerate(row):
            if _ != i:
                bonds += 2*bond
        return bonds
        
    def GetElectronInfo(self):
        self.lps = []
        self.rads = [] 
        for i in range(len(self.yarpecule.bond_mats)):
            self.lps += [[]]
            self.rads += [[]]
        for ind,bm in enumerate(self.yarpecule.bond_mats):
            for i in range(bm.shape[0]):
                self.lps[ind].append(bm[i,i])
                self.rads[ind].append(self.EncodeRadicalInfo(bm,i))
    
    def GetResonanceInfo(self):
        bond_tensor = np.stack(self.yarpecule.bond_mats, axis=0)
        n_atoms = bond_tensor.shape[1]
        # Initialize dictionary to store unique values
        unique_diagonals = {}
        unique_bonds = {}
        for i in range(n_atoms):
            # Diagonal (atom-specific info)
            diagcheck = len(np.unique(bond_tensor[:, i, i]))
            if diagcheck == 1:
                unique_diagonals[i] = [1,0]
            else:
                unique_diagonals[i] = [0,1]
            for j in range(i + 1, n_atoms):  # off-diagonal pairs only once (i < j)
                # Bonds between atom i and j
                values = bond_tensor[:, i, j]
                unique_vals = np.unique(values)
                if unique_vals.size > 0:
                    check = len(unique_vals)
                    if check == 1:
                        unique_bonds[(i, j)] = [1,0]
                    else:
                        unique_bonds[(i, j)] = [0,1]
        return unique_diagonals,unique_bonds

    def UniqueMats(self):
        # Stack into 3D array: shape (num_structures, n_atoms, n_atoms)
        bond_tensor = np.stack(self.yarpecule.bond_mats, axis=0)

        # Flatten each matrix to 1D and view as rows in a 2D array
        flattened = bond_tensor.reshape(len(bond_tensor), -1)

        # Use np.unique across rows to get unique bond matrices
        unique_bond_mats = np.unique(flattened, axis=0)

        # Number of unique bond matrices
        num_unique = unique_bond_mats.shape[0]    
        return num_unique
    
    def run(self):
        self.GetElectronInfo()
        self.atom_resonance,self.bond_resonance = self.GetResonanceInfo()
        self.unique_mats = self.UniqueMats()



