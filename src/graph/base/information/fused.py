from ..base import BaseFeaturizer
from rdkit import Chem
from dataclasses import dataclass

@dataclass
class FusedInformationParams:
    getfusedinformation: bool = False

class FusedInformation(BaseFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
    

    def ObtainFusedRingInfo(self):
        mol = self.matrixdescriptors.new_mol
        if mol is None:
            raise ValueError("Invalid SMILES string")

        ring_info = mol.GetRingInfo()
        atom_ring_counts = {}
        bond_ring_counts = {}

        for atom_idx in range(mol.GetNumAtoms()):
            atom_ring_counts[atom_idx] = ring_info.NumAtomRings(atom_idx)

        for bond_idx in range(mol.GetNumBonds()):
            bond_ring_counts[bond_idx] = ring_info.NumBondRings(bond_idx)

        self.fusedringinfo = {
            "atom_ring_counts": atom_ring_counts,
            "bond_ring_counts": bond_ring_counts,
        }

    def AtominFusedRing(self,ind):
        """
        Check if an atom is in a fused ring system.
        
        Args:
            ind (int): Index of the atom.
        
        Returns:
            bool: True if the atom is in a fused ring system, False otherwise.
        """

        if self.params.getfusedinformation:
            if self.fusedringinfo is None:
                return [1,0]
            atom_ring_counts = self.fusedringinfo["atom_ring_counts"]
        
            if atom_ring_counts.get(ind, 0) > 1:
                return [0,1]
            else:
                return [1,0]
        else:
            return []
    
    def AtominXRings(self,ind):
        """
        Check if an atom is in a fused ring system.
        
        Args:
            ind (int): Index of the atom.
        
        Returns:
            bool: True if the atom is in a fused ring system, False otherwise.
        """
        if self.params.getfusedinformation:

            if self.fusedringinfo is None:
                raise ValueError("Fused ring information not available. Call ObtainFusedRingInfo() first.")
            
            atom_ring_counts = self.fusedringinfo["atom_ring_counts"]
            return [atom_ring_counts.get(ind, 0)]
        else:
            return []
        

    def BondinFusedRing(self,edge):
        """
        Check if a bond is in a fused ring system.
        
        Args:
            edge (tuple): Tuple containing the indices of the two atoms forming the bond.
        
        Returns:
            bool: True if the bond is in a fused ring system, False otherwise.
        """

        if self.params.getfusedinformation:
            if self.fusedringinfo is None:
                raise ValueError("Fused ring information not available. Call ObtainFusedRingInfo() first.")
            
            bond_ring_counts = self.fusedringinfo["bond_ring_counts"]
            if bond_ring_counts.get(tuple(edge), 0) > 1:
                return [0,1]
            else:
                return [1,0]
        else:
            return []
        