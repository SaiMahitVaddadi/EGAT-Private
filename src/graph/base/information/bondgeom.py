from ..base import BaseFeaturizer
from dataclasses import dataclass
import numpy as np
from rdkit import Chem

@dataclass
class BondGeometryParams:
    """
    Parameters
    ----------
    getbondlength : bool, optional
        A flag indicating whether to calculate bond lengths. Default is False.
    getatomdistance : bool, optional
        A flag indicating whether to calculate atom distances. Default is False.
    getbondangle : str, optional
        Specifies the type of bond angle to calculate. Options are 'first', 'smallest', 
        'largest', 'average', 'all', or 'main'. Default is None.
    getdihedral : str, optional
        Specifies the type of dihedral angle to calculate. Options are 'first', 
        'firstsmallest', 'firstlargest', 'firstaverage', 'firstall', 'firstmain', 
        'smallest', 'largest', 'average', 'all', or 'main'. Default is None.
    getbondmidpoint : bool, optional
        A flag indicating whether to calculate bond midpoints. Default is False.
    """
    getbondlength: bool = False
    getatomdistance: bool = False
    getbondangle: str = None
    getdihedral: str = None
    getbondmidpoint: bool = False

class BondGeometryInformation(BaseFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
        
    def BondLength(self,edge,id=1):
        if self.params.getbondlength:
            mol = self.conformer

            
            try:
                bond = mol.GetBondBetweenAtoms(edge[0], edge[1])
                atom1 = bond.GetBeginAtomIdx()
                atom2 = bond.GetEndAtomIdx()
            except:
                atom1 = edge[0]
                atom2 = edge[1]
            pos1 = mol.GetConformer().GetAtomPosition(atom1)
            pos2 = mol.GetConformer().GetAtomPosition(atom2)
            bond_length = pos1.Distance(pos2)
            return [bond_length]
        else:
            return []

    def AtomDistance(self, edge,id=1):
        if self.params.getatomdistance:
            mol = self.conformer
            atom1 = edge[0]
            atom2 = edge[1]
            pos1 = mol.GetConformer().GetAtomPosition(atom1)
            pos2 = mol.GetConformer().GetAtomPosition(atom2)
            distance = pos1.Distance(pos2)
            return [distance]
        else:
            return []

    def GetAngle(self,atom1, atom2,id=1):
        angles = []
        mol = self.conformer
        for neighbor in np.nonzero(self.matrixdescriptors.adj_mat[atom2])[0]:
            angle = mol.GetBondBetweenAtoms(atom1, atom2).GetAngle(mol.GetConformer(), atom1, atom2, neighbor)
            angles.append(angle)
        return angles 

    def BondAngle(self,edge,id=1):
        if self.params.getbondangle == 'first': 
            atom1 = edge[0]
            atom2 = edge[1]
            neighbors = np.nonzero(self.matrixdescriptors.adj_mat[atom2])[0]
            atom3 = neighbors[neighbors != atom1][0] if len(neighbors[neighbors != atom1]) > 0 else None
            mol = self.conformer
            try: 
                angle = mol.GetBondBetweenAtoms(atom1, atom2).GetAngle(mol.GetConformer(), atom1, atom2, atom3)
                return [angle]
            except:
                return [0]
        elif self.params.getbondangle == 'smallest':
            angles = self.GetAngle(atom1,atom2,id)
            return [min(angles)]
        elif self.params.getbondangle == 'largest':
            angles = self.GetAngle(atom1,atom2,id)
            return [max(angles)]
        elif self.params.getbondangle == 'average':
            angles = self.GetAngle(atom1,atom2,id)
            return [sum(angles)/len(angles)]
        elif self.params.getbondangle == 'all':
            angles = self.GetAngle(atom1,atom2,id)
            return self.FlattenList(angles)
        elif self.params.getbondangle == 'main':
            angles = self.GetAngle(atom1,atom2,id)
            return [min(angles),max(angles),sum(angles)/len(angles)]
        else:
            return []

    def GetDihedral(self, edge,id=1):
        atom1 = edge[0]
        atom2 = edge[1]
        dihedrals = []
        mol = self.conformer
        for neighbor1 in np.nonzero(self.matrixdescriptors.adj_mat[int(atom2)])[0]:
            row = []
            for neighbor2 in np.nonzero(self.matrixdescriptors.adj_mat[int(neighbor1)])[0]:
                if int(neighbor1) != int(neighbor2):
                    dihedral = Chem.rdMolTransforms.GetDihedralDeg(mol.GetConformer(), int(atom1), int(atom2), int(neighbor1), int(neighbor2))
                    row.append(dihedral)
            dihedrals.append(row)
        return dihedrals
    
    def FlattenList(self, nested_list):
        return [item for sublist in nested_list for item in sublist]

    def DihedralAngle(self,edge,id=1):
        if self.params.getdihedral == 'first':
            atom1 = edge[0]
            atom2 = edge[1]
            neighbors_atom2 = np.nonzero(self.matrixdescriptors.adj_mat[atom2])[0]
            atom3 = next((neighbor for neighbor in neighbors_atom2 if neighbor != atom1), None)
            if atom3 is not None:
                neighbors_atom3 = np.nonzero(self.matrixdescriptors.adj_mat[atom3])[0]
                atom4 = next((neighbor for neighbor in neighbors_atom3 if neighbor != atom2), None)
            else:
                atom4 = None
            mol = self.conformer
            try:
                dihedral = Chem.rdMolTransforms.GetDihedralDeg(mol.GetConformer(), int(atom1), int(atom2), int(atom3), int(atom4))
                return [dihedral]
            except:
                return [0]
        elif self.params.getdihedral == 'firstsmallest':
            dihedrals = self.GetDihedral(edge)
            return [min(dihedrals[0])]
        elif self.params.getdihedral == 'firstlargest':
            dihedrals = self.GetDihedral(edge)
            return [max(dihedrals[0])]
        elif self.params.getdihedral == 'firstaverage': 
            dihedrals = self.GetDihedral(edge)
            return [sum(dihedrals[0])/len(dihedrals[0])]
        elif self.params.getdihedral == 'firstall':
            dihedrals = self.GetDihedral(edge)
            return dihedrals[0]
        elif self.params.getdihedral == 'firstmain':
            dihedrals = self.GetDihedral(edge)
            return [min(dihedrals[0]),max(dihedrals[0]),sum(dihedrals[0])/len(dihedrals[0])]
        elif self.params.getdihedral == 'smallest':
            dihedrals = self.GetDihedral(edge)
            return [min(self.FlattenList(dihedrals))]
        elif self.params.getdihedral == 'largest':
            dihedrals = self.GetDihedral(edge)
            return [max(self.FlattenList(dihedrals))]
        elif self.params.getdihedral == 'average': 
            dihedrals = self.GetDihedral(edge)
            return [sum(self.FlattenList(dihedrals))/len(self.FlattenList(dihedrals))]
        elif self.params.getdihedral == 'all':
            dihedrals = self.GetDihedral(edge)
            return self.FlattenList(dihedrals)
        elif self.params.getdihedral == 'main':
            dihedrals = self.GetDihedral(edge)
            return [min(self.FlattenList(dihedrals)),max(self.FlattenList(dihedrals)),sum(self.FlattenList(dihedrals))/len(self.FlattenList(dihedrals))]
        else:
            return []
    
    def BondMidpoint(self, edge,id=1):
        if self.params.getbondmidpoint:
            atom1 = edge[0]
            atom2 = edge[1]
            mol = self.conformer
            pos1 = mol.GetConformer().GetAtomPosition(atom1)
            pos2 = mol.GetConformer().GetAtomPosition(atom2)
            midpoint = (pos1 + pos2) / 2
            return [midpoint.x, midpoint.y, midpoint.z]
        else:
            return []
       
