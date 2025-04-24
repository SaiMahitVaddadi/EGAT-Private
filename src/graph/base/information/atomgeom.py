from ..base import BaseFeaturizer
from rdkit import Chem
import cctk 
from scipy.spatial import ConvexHull
import numpy as np
import pandas as pd 
from dataclasses import dataclass
from rdkit.Chem import rdMolDescriptors,AllChem, rdFreeSASA
import math
from rdkit.Geometry import Point3D
from collections import defaultdict

#To-DO : Add The Steric Hindrance Index And Ionization Potential Index



@dataclass
class AtomGeometryParams:
    """
    Parameters
    ----------
    removecoordinationinfo : bool, optional
        A flag indicating whether to remove coordination information. 
        Default is False.
    getdistancetocenterofmass : bool, optional
        A flag indicating whether to calculate the distance to the center of mass. 
        Default is False.
    getsterichindrance : bool, optional
        A flag indicating whether to calculate steric hindrance. 
        Default is False.
    getasa : bool, optional
        A flag indicating whether to calculate atomic solvent accessibility. 
        Default is False.
    getgaussiancurvature : bool, optional
        A flag indicating whether to calculate Gaussian curvature. 
        Default is False.
    getmolecularshapeindex : bool, optional
        A flag indicating whether to calculate the molecular shape index. 
        Default is False.
    getdistancetoconvexhull : bool, optional
        A flag indicating whether to calculate the distance to the convex hull. 
        Default is False.
    """
    removecoordinationinfo: bool = False
    getdistancetocenterofmass: bool = False
    getsterichindrance: bool = False
    getasa: bool = False
    getgaussiancurvature: bool = False
    getmolecularshapeindex: bool = False
    getdistancetoconvexhull: bool = False
    getburiedvolume: bool = False

class AtomGeometryInformation(BaseFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
        self.vdw_radii = {
            'H': 1.20, 'He': 1.40, 'Li': 1.82, 'Be': 1.53, 'B': 1.92, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47, 'Ne': 1.54,
            'Na': 2.27, 'Mg': 1.73, 'Al': 1.84, 'Si': 2.10, 'P': 1.80, 'S': 1.80, 'Cl': 1.75, 'Ar': 1.88, 'K': 2.75, 'Ca': 2.31,
            'Sc': 2.11, 'Ti': 2.00, 'V': 2.00, 'Cr': 2.00, 'Mn': 2.00, 'Fe': 2.00, 'Co': 2.00, 'Ni': 1.63, 'Cu': 1.40, 'Zn': 1.39,
            'Ga': 1.87, 'Ge': 2.11, 'As': 1.85, 'Se': 1.90, 'Br': 1.85, 'Kr': 2.02, 'Rb': 3.03, 'Sr': 2.49, 'Y': 2.00, 'Zr': 2.00,
            'Nb': 2.00, 'Mo': 2.00, 'Tc': 2.00, 'Ru': 2.00, 'Rh': 2.00, 'Pd': 1.63, 'Ag': 1.72, 'Cd': 1.58, 'In': 1.93, 'Sn': 2.17,
            'Sb': 2.00, 'Te': 2.06, 'I': 1.98, 'Xe': 2.16, 'Cs': 3.43, 'Ba': 2.68, 'La': 2.00, 'Ce': 2.00, 'Pr': 2.00, 'Nd': 2.00,
            'Pm': 2.00, 'Sm': 2.00, 'Eu': 2.00, 'Gd': 2.00, 'Tb': 2.00, 'Dy': 2.00, 'Ho': 2.00, 'Er': 2.00, 'Tm': 2.00, 'Yb': 2.00,
            'Lu': 2.00, 'Hf': 2.00, 'Ta': 2.00, 'W': 2.00, 'Re': 2.00, 'Os': 2.00, 'Ir': 2.00, 'Pt': 1.75, 'Au': 1.66, 'Hg': 1.55,
            'Tl': 1.96, 'Pb': 2.02, 'Bi': 2.07, 'Po': 2.00, 'At': 2.00, 'Rn': 2.00, 'Fr': 2.00, 'Ra': 2.00, 'Ac': 2.00, 'Th': 2.00,
            'Pa': 2.00, 'U': 1.86, 'Np': 2.00, 'Pu': 2.00, 'Am': 2.00, 'Cm': 2.00, 'Bk': 2.00, 'Cf': 2.00, 'Es': 2.00, 'Fm': 2.00,
            'Md': 2.00, 'No': 2.00, 'Lr': 2.00, 'Rf': 2.00, 'Db': 2.00, 'Sg': 2.00, 'Bh': 2.00, 'Hs': 2.00, 'Mt': 2.00, 'Ds': 2.00,
            'Rg': 2.00, 'Cn': 2.00, 'Nh': 2.00, 'Fl': 2.00, 'Mc': 2.00, 'Lv': 2.00, 'Ts': 2.00, 'Og': 2.00
        }
        
    

    def GrabConformer(self,id=1):
        if id == 1: return self.matrixdescriptors.new_mol
        else:
            return self.matrixdescriptors.new_mol.GetConformer(conf_id=id)
    

    def GrabGeometry(self,id=1):
        mol = self.GrabConformer(id)
        self.positions = {}
        for atom in mol.GetAtoms():
            pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
            atom_map_num = atom.GetAtomMapNum()
            ind = atom.GetIdx()
            self.positions[ind] = [pos.x, pos.y, pos.z]
        return mol 
    
    def CalcASA(self,id=1):
        mol = self.GrabConformer(id)
        radii = rdFreeSASA.ClassifyAtoms(mol)
        total_asa = rdFreeSASA.CalcASA(mol, radii)
        # Get total TPSA
        total_tpsa = rdMolDescriptors.CalcTPSA(mol)
        self.asa = []
        self.percent_asa = []
        self.percent_asa_wrt_atom = []
        # Get per-atom TPSA contributions
        self.tpsa = rdMolDescriptors._CalcTPSAContribs(mol)  # returns a list of floats
        self.percent_tpsa = list(np.array(self.tpsa)/total_tpsa * 100)
        self.percent_tpsa_wrt_atom = []
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            asa = float(mol.GetProp(f"_FreeSASA_atom_{idx}"))
            percent = (asa / total_asa) * 100 if total_asa > 0 else 0
            self.asa.append(asa)
            self.percent_asa.append(percent)
            # Radius from FreeSASA
            r = radii[idx]
            full_sa = 4 * math.pi * r ** 2 if r > 0 else 1e-10
            self.percent_asa_wrt_atom.append(asa / full_sa * 100 if full_sa > 0 else 0)
            self.percent_tpsa_wrt_atom.append(self.percent_tpsa[idx]/full_sa * 100)

    def BuriedVolumecalc(self,center_type="centroid", center_idx=None, center_coords=None, R=3.5, grid_step=0.1, id=1):
        mol = self.GrabGeometry(id)
        atom_coords = pd.DataFrame(self.positions).T.values
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        radii = [self.vdw_radii.get(s, 2.00) for s in symbols]

        # Determine center
        if center_type == "atom":
            center = atom_coords[center_idx]
        elif center_type == "coords":
            center = np.array(center_coords)
        else:  # centroid
            center = np.mean(atom_coords, axis=0)

        # Voxel grid setup
        grid_range = np.arange(-R, R + grid_step, grid_step)
        buried_voxels = defaultdict(int)
        total_voxels = 0

        for dx in grid_range:
            for dy in grid_range:
                for dz in grid_range:
                    point = center + np.array([dx, dy, dz])
                    if np.linalg.norm(point - center) <= R:
                        total_voxels += 1
                        for i, (coord, r) in enumerate(zip(atom_coords, radii)):
                            if np.linalg.norm(point - coord) <= r:
                                buried_voxels[i] += 1
                                break  # Only count one atom per point

        # Compute per-atom and total %Vbur
        total_buried = sum(buried_voxels.values())
        percent_buried_total = 100 * total_buried / total_voxels

        self.bv = []
        self.bvcontrib = []
        self.bvratio = []
        for i in range(mol.GetNumAtoms()):
            vdw_r = radii[i]
            atom_volume = (4/3) * math.pi * vdw_r**3
            buried_voxel_count = buried_voxels[i]
            voxel_volume = grid_step ** 3
            buried_volume = buried_voxel_count * voxel_volume
            contribution = 100 * buried_volume / (total_voxels * voxel_volume) if total_voxels > 0 else 0
            burial_ratio = buried_volume / atom_volume if atom_volume > 0 else 0
            self.bv.append(buried_volume)
            self.bvcontrib.append(contribution)
            self.bvratio.append(burial_ratio)

        



    def AtomCoordination(self,ind,id=1):
        if not self.params.removecoordinationinfo:
            self.GrabGeometry(id)
            return self.positions[ind] # Grab the conforem
        else:
            return []


    def toCCTK(self,id=1):
        mol = self.GrabGeometry(id)
        atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        geometry = pd.DataFrame(self.positions).T.values
        cctk_molecule = cctk.Molecule(atomic_numbers=atomic_numbers, geometry=geometry)
        return mol, cctk_molecule

        
    def DistanceToCenterOfMass(self, ind,id=1):
        if self.params.getdistancetocenterofmass:
            rdmolecule,cctkmolecule = self.toCCTK(id)
            com = cctkmolecule.center_of_mass()
            position = self.positions[ind]
            distance = np.linalg.norm(position-com)
            return [distance]
        else:
            return []

    def StericHindrance(self, ind,id=1):
        if self.params.getsterichindrance:
            mol = self.GrabGeometry(id)
            position = self.positions[ind]
            adj_matrix = Chem.GetAdjacencyMatrix(mol)
            hindrance = 0
            for neighbor in range(mol.GetNumAtoms()):
                if neighbor != ind:
                    npos = mol.GetConformer().GetAtomPosition(neighbor)
                    nposition = np.array([npos.x, npos.y, npos.z])
                    distance = np.linalg.norm(npos-position)  # Calculate Euclidean distance
                    if distance < 3.5:
                        if adj_matrix[ind][neighbor] == 0:
                            hindrance += 1 / distance
                        elif adj_matrix[ind][neighbor] == 1:
                            hindrance += 1/ (distance **.5)
            return [hindrance]
        else:
            return []
        
    def VdWStrain(self, ind,id=1):
        if self.params.getsterichindrance:
            mol = self.GrabConformer(id)
            pos = mol.GetConformer().GetAtomPosition(ind)
            adj_matrix = Chem.GetAdjacencyMatrix(mol)
            position = np.array([pos.x, pos.y, pos.z])
            hindrance = 0
            for neighbor in range(mol.GetNumAtoms()):
                if neighbor != ind:
                    npos = mol.GetConformer().GetAtomPosition(neighbor)
                    nposition = np.array([npos.x, npos.y, npos.z])
                    distance = np.linalg.norm(npos-position)  # Calculate Euclidean distance
                    if distance < self.vdw_radii[self.matrixdescriptors.element[ind]] + self.vdw_radii[self.matrixdescriptors.element[neighbor]]:
                        if adj_matrix[ind][neighbor] == 0:
                            hindrance += 1 / distance
                        elif adj_matrix[ind][neighbor] == 1:
                            hindrance += 1 / (distance **.5)

            return [hindrance]
        else:
            return []
    

    def getVdWSurfaceArea(self,ind):
        radius = self.vdw_radii[self.matrixdescriptors.element[ind]]  # Van der Waals radius of the atom
        exposed_area = 4 * np.pi * radius**2  # Start with full surface area of the sphere
        vdw_surface_area = exposed_area
        return vdw_surface_area
    
    def getVdWMol(self,ind):
        radius = self.vdw_radii[self.matrixdescriptors.element[ind]]  # Van der Waals radius of the atom
        exposed_area = 4/2 * np.pi * radius**3  # Start with full surface area of the sphere
        vdw_surface_area = exposed_area
        return vdw_surface_area


    def AtomicSolventAccessibility(self, ind,id=1):
        if self.params.getasa:
            return [self.asa[ind],self.percent_asa[ind],self.percent_asa_wrt_atom[ind],self.tpsa[ind],self.percent_tpsa[ind],self.percent_tpsa_wrt_atom[ind]]
        else:
            return []
    
    def GaussianCurvature(self, ind,id=1):
        if self.params.getgaussiancurvature:
            mol = self.GrabConformer(id)
            pos = mol.GetConformer().GetAtomPosition(ind)
        
            # Calculate Gaussian curvature using neighboring atoms
            neighbors = np.nonzero(self.matrixdescriptors.adj_mat[ind])[0]
            if len(neighbors) < 3:
                return [0]  # Not enough neighbors to define a surface
            angles = []
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    pos1 = mol.GetConformer().GetAtomPosition(int(neighbors[i]))
                    pos2 = mol.GetConformer().GetAtomPosition(int(neighbors[j]))
                    v1 = np.array([pos1.x - pos.x, pos1.y - pos.y, pos1.z - pos.z])
                    v2 = np.array([pos2.x - pos.x, pos2.y - pos.y, pos2.z - pos.z])
                    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Clip to handle numerical errors
                    angles.append(angle)
            gaussian_curvature = 2 * np.pi - sum(angles)
            return [gaussian_curvature]
        else:
            return []

    def MolecularShapeIndex(self, ind,id=1):
        if self.params.getmolecularshapeindex:
            mol = self.GrabConformer(id)
            pos = mol.GetConformer().GetAtomPosition(ind)
        
           # Calculate Gaussian curvature using neighboring atoms
            neighbors = np.nonzero(self.matrixdescriptors.adj_mat[ind])[0]
            if len(neighbors) < 3:
                return [0]  # Not enough neighbors to define a surface
            distances = []
            for neighbor in neighbors:
                pos = mol.GetConformer().GetAtomPosition(int(neighbor))
                distance = np.linalg.norm(np.array([pos.x, pos.y, pos.z]) - np.array([self.positions[ind][0], self.positions[ind][1], self.positions[ind][2]]))
                distances.append(distance)
            shape_index = sum(distances) / len(distances)
            return [shape_index]
        else:
            return []

    def DistanceToConvexHull(self, ind,id=1):
        if self.params.getdistancetoconvexhull:
            mol = self.GrabConformer(id)
            atom_position = mol.GetConformer().GetAtomPosition(ind)
            
            # Calculate convex hull of the molecule
            atom_positions = [mol.GetConformer().GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
            points = np.array([[pos.x, pos.y, pos.z] for pos in atom_positions])
            hull = ConvexHull(points)
            
            # Calculate distance from atom to convex hull
            distances = []
            for simplex in hull.simplices:
                vertices = points[simplex]
                distance = np.min(np.linalg.norm(vertices - np.array([atom_position.x, atom_position.y, atom_position.z]), axis=1))
                distances.append(distance)
            return [min(distances)]
        else:
            return []

    def GetVanDerWaalsRadii(self, ind):
        element = self.matrixdescriptors.element[ind]
        return [self.vdw_radii.get(element, 2.00)]  # Default to 2.00 if element not found

    
    def GetBuriedVolume(self,ind,id=1):
        if self.params.getburiedvolume:
            return [self.bv[ind],self.bvcontrib[ind],self.bvratio[ind]]
        else:
            return []
        

    def SurroundingVdWMetrics(self,ind,id=1):
        mol = self.matrixdescriptors.new_mol
        neighbors = [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(ind).GetNeighbors()]
        sub_sa = 0
        sub_vol = 0
        sub_sa_ratio = 0
        sub_vol_ratio = 0
        for neighbor_idx in neighbors:
            sub_sa += self.getVdWSurfaceArea(mol.GetAtomWithIdx(neighbor_idx).GetSymbol())
            sub_vol += self.getVdWMol(mol.GetAtomWithIdx(neighbor_idx).GetSymbol())
        sub_sa_ratio = sub_sa / self.getVdWSurfaceArea(mol.GetAtomWithIdx(ind).GetSymbol())
        sub_vol_ratio = sub_vol / self.getVdWMol(mol.GetAtomWithIdx(ind).GetSymbol())
        return [sub_sa,sub_vol,sub_sa_ratio,sub_vol_ratio]
    
    def SurroundingIPMetrics(self,ind,id=1):
        pass


