import numpy as np
from ..base import BaseFeaturizer

from dataclasses import dataclass, field
from typing import Dict, Optional
@dataclass
class SterimolParams:
    getsterimol: bool = False
    local_cutoff: float = 4.0
    use_vdw: bool = False
    
class SterimolFeaturizer(BaseFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)

    def local_sterimol(self,bond_atom1, bond_atom2, id=1, local_cutoff=4.0, use_vdw=False):
        """
        Calculate Sterimol parameters localized around a specific bond defined by two atoms.
        
        Args:
          mol (rdkit.Chem.Mol): Molecule loaded from a structure file.
          bond_atom1, bond_atom2 (int): Atom indices (0-indexed) defining the bond.
          local_cutoff (float): Distance (in Angstroms) to include atoms in the local environment.
          use_vdw (bool): Whether to add van der Waals radii.
          radii_dict (dict): Dictionary mapping atomic numbers to van der Waals radii.
          
        Returns:
          tuple: (L_local, B1_local, B5_local)
        """
        mol = self.conformer
        coords = np.array(list(self.positions.values()))
        # Define the local bond axis (from atom1 to atom2)
        origin = coords[bond_atom1]
        axis_vector = coords[bond_atom2] - origin
        unit_axis = axis_vector / np.linalg.norm(axis_vector)
        
        # Identify local atoms: those within local_cutoff distance from either atom on bond_atom1 side.
        # One simple way: include atoms for which the distance to bond_atom1 is within cutoff
        local_indices = [i for i in range(mol.GetNumAtoms())
                        if np.linalg.norm(coords[i] - origin) <= local_cutoff]
        
        L_values = []
        perp_values = []

        radii_dict = {
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
        
        for i in local_indices:
            rel_vec = coords[i] - origin
            # Projection along the bond axis
            proj = np.dot(rel_vec, unit_axis)
            # Perpendicular component
            perp_vec = rel_vec - proj * unit_axis
            dist_perp = np.linalg.norm(perp_vec)
            
            # Optionally adjust for van der Waals radii:
            if use_vdw:
                atom = mol.GetAtomWithIdx(i)
                vdw = radii_dict.get(atom.GetSymbol(), 1.5)
                dist_perp += vdw
            
            L_values.append(proj)
            perp_values.append(dist_perp)
        
        # L_local is the maximum extension along the local axis
        L_local = max(L_values)
        B1_local = min(perp_values)  # minimal perpendicular distance
        B5_local = max(perp_values)  # maximal perpendicular distance
        # Calculate the direction of the bond axis
        direction = unit_axis
        return [L_local, B1_local, B5_local],direction
    
    def StermiolMatrices(self,id=1):
        """
        Calculate Sterimol parameters for all atoms in the molecule.
        
        Returns:
          list: List of Sterimol parameters for each atom.
        """
        mol = self.matrixdescriptors.new_mol
        sterimol_params = dict()
        
        for atom in mol.GetAtoms():
            # Get the indices of the two atoms in the bond
            bonds = atom.GetBonds()
            if len(bonds) > 0:
                bond_atom1 = bonds[0].GetBeginAtomIdx()
                bond_atom2 = bonds[0].GetEndAtomIdx()
                sterimols, direction = self.local_sterimol(bond_atom1, bond_atom2,id,local_cutoff=self.params.local_cutoff,use_vdw=self.params.use_vdw)
                sterimol_params[tuple([bond_atom1,bond_atom2])] = sterimols
        
        self.sterimoldata = sterimol_params
    

    def BondSterimolFeatures(self,edge,id=1):
        if self.params.getsterimol:
            return self.sterimoldata[tuple(edge)]
        else:
            return []
  
    def AtomSterimolFeatures(self,ind,id=1):
        if self.params.getsterimol:
            neighbors = [neighbor.GetIdx() for neighbor in self.conformer.GetAtomWithIdx(ind).GetNeighbors()]
            sterimol_features = []
            for neighbor in neighbors:
                try:
                    sterimol_features += self.sterimoldata[tuple([ind,neighbor])]
                except:
                    sterimol_features += self.sterimoldata[tuple([neighbor,ind])]
            sterimol_features = np.array(sterimol_features)
            try:
                return list(sterimol_features.sum(axis=1))
            except:
                return list(sterimol_features)
        else:
            return []

    def MomentDescriptors(self,ind,id=1,geom = True,sterimol=True,unitvec=False):
        """
        vectors: numpy array of shape (n_vectors, 6)
        normalize: whether to normalize each feature to zero mean and unit variance
        Returns: dictionary of scalar descriptors
        """
        vectors = []
        neighbors = [neighbor.GetIdx() for neighbor in self.conformer.GetAtomWithIdx(ind).GetNeighbors()]
        for neighbor in neighbors:
            data = [] 
            coords = np.array(list(self.positions.values()))
            # Define the local bond axis (from atom1 to atom2)
            origin = coords[ind]
            axis_vector = coords[neighbor] - origin
            if geom:
                if unitvec: data += list(axis_vector/np.linalg.norm(axis_vector))
                else: data += list(axis_vector)

            if sterimol:
                try:
                    data += self.sterimoldata[tuple([ind,neighbor])]
                except:
                    data += self.sterimoldata[tuple([neighbor,ind])]
            vectors.append(np.array(data))        
        vectors = np.array(vectors)

        _, n_dim = vectors.shape
        
        # Compute moment of inertia tensor
        M = np.zeros((n_dim, n_dim))
        for v in vectors:
            M += np.outer(v, v)

        # Eigen decomposition
        eigenvalues = np.linalg.eigvalsh(M)  # eigvalsh = faster for symmetric matrices

        # Sort eigenvalues in descending order
        eigenvalues = np.sort(eigenvalues)[::-1]

        # Compute descriptors
        trace = np.sum(eigenvalues)
        determinant = np.prod(eigenvalues)
        frobenius_norm = np.sqrt(np.sum(eigenvalues**2))
        anisotropy = (eigenvalues[0] - eigenvalues[-1]) / trace if trace != 0 else 0
        variance = np.var(eigenvalues)

        # Optional: entropy-like descriptor (spreadness)
        eigenvalues_normalized = eigenvalues / trace if trace != 0 else np.ones_like(eigenvalues) / len(eigenvalues)
        entropy = -np.sum(eigenvalues_normalized * np.log(eigenvalues_normalized + 1e-12))  # small epsilon for stability

        return [trace, determinant, frobenius_norm, anisotropy, variance, entropy]


    def GeomMomentDescriptors(self,ind,id=1):
        """
        Calculate geometric moment descriptors for a given atom index.
        
        Args:
          ind (int): Atom index.
          
        Returns:
          list: List of geometric moment descriptors.
        """
        if 'geometric' in self.params.getmomentdescriptors:
            return self.MomentDescriptors(ind,geom=True,sterimol=False,unitvec=False)
        elif 'geometricunitvec' in self.params.getmomentdescriptors:
            return self.MomentDescriptors(ind,geom=True,sterimol=False,unitvec=True)
        else:
            return []
    
    def SterimolMomentDescriptors(self,ind,id=1):
        """
        Calculate Sterimol moment descriptors for a given atom index.
        
        Args:
          ind (int): Atom index.
          
        Returns:
          list: List of Sterimol moment descriptors.
        """
        if 'sterimol' in self.params.getmomentdescriptors:
            return self.MomentDescriptors(ind,geom=False,sterimol=True,unitvec=False)
        else:
            return []
    
    def SterGeomMomentDescriptors(self,ind,id=1):
        """
        Calculate Sterimol and geometric moment descriptors for a given atom index.
        
        Args:
          ind (int): Atom index.
          
        Returns:
          list: List of Sterimol and geometric moment descriptors.
        """
        if 'sterimol' in self.params.getmomentdescriptors:
            if 'geometric' in self.params.getmomentdescriptors:
                return self.MomentDescriptors(ind,geom=True,sterimol=True,unitvec=False)
            elif 'geometricunitvec' in self.params.getmomentdescriptors:
                return self.MomentDescriptors(ind,geom=True,sterimol=True,unitvec=True)
        else:
            return []
        
    
    def SolidAngleCoverage(self,ind,id=1):
        """
        Calculate the solid angle coverage for a given atom index.
        
        Args:
          ind (int): Atom index.
          
        Returns:
          dict: Dictionary containing solid angle coverage and fraction of sphere.
        """
        if self.params.getsolidanglecoverage:
            coords = np.array(list(self.positions.values()))
            neighbors = [neighbor.GetIdx() for neighbor in self.conformer.GetAtomWithIdx(ind).GetNeighbors()]
            vectors = []
            for neighbor in neighbors:
                vectors.append(coords[neighbor])
            vectors = np.array(vectors)
            return compute_solid_angle_coverage(vectors)
        else:
            return []
        



def solid_angle_of_triangle(a, b, c):
    """
    a, b, c: 3 unit vectors forming a spherical triangle
    Returns the solid angle subtended at the origin (in steradians)
    """

    # Ensure all vectors are unit length
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    c = c / np.linalg.norm(c)

    # Compute edge lengths (chordal distances)
    ab = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
    bc = np.arccos(np.clip(np.dot(b, c), -1.0, 1.0))
    ca = np.arccos(np.clip(np.dot(c, a), -1.0, 1.0))

    # Compute the spherical excess using L'Huilier's formula
    s = 0.5 * (ab + bc + ca)
    tan_e_over4 = np.sqrt(np.tan(s/2) * np.tan((s-ab)/2) * np.tan((s-bc)/2) * np.tan((s-ca)/2))
    solid_angle = 4 * np.arctan(tan_e_over4)

    return solid_angle

def compute_solid_angle_coverage(vectors):
    """
    vectors: numpy array of shape (n_vectors, 3) - only x,y,z components
    Returns total solid angle covered (in steradians) and as a fraction of 4pi
    """

    n_vectors, n_dim = vectors.shape
    assert n_dim == 3, "Input must be (n_vectors, 3)."

    # Normalize all vectors to unit sphere
    unit_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    # Very simple: if only 3 vectors, assume they form one triangle
    if n_vectors == 3:
        omega = solid_angle_of_triangle(unit_vectors[0], unit_vectors[1], unit_vectors[2])
    elif n_vectors > 3:
        # If more vectors: approximate using convex hull
        from scipy.spatial import ConvexHull

        hull = ConvexHull(unit_vectors)
        omega = 0
        for simplex in hull.simplices:
            i, j, k = simplex
            omega += solid_angle_of_triangle(unit_vectors[i], unit_vectors[j], unit_vectors[k])
    else:
        omega = 0

    # Normalize by 4pi if desired
    fraction = omega / (4 * np.pi)

    return [omega, fraction]
