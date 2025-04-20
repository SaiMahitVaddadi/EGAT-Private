import numpy as np
from ..base import BaseFeaturizer

class SterimolFeaturizer(BaseFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)

    def GrabConformer(self,id=1):
        if id == 1: return self.matrixdescriptors.new_mol
        else:
            return self.matrixdescriptors.new_mol.GetConformer(conf_id=id)
    

    def local_sterimol(self,bond_atom1, bond_atom2, id=1, local_cutoff=4.0, use_vdw=False, radii_dict=None):
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
        conf = self.GrabConformer(id)
        mol = self.matrixdescriptors.new_mol
        coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
        
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
                vdw = radii_dict.get(atom.GetAtomicNum(), 1.5)
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
    
  
    def atom_sterimol(self,atom):
        """
        Calculate Sterimol parameters for a single atom.
        
        Args:
          atom (rdkit.Chem.Atom): Atom object from RDKit.
          
        Returns:
          tuple: (L, B1, B5)
        """
        # Get the atomic number
        atomic_num = atom.GetAtomicNum()
        
        # Define Sterimol parameters based on atomic number
        if atomic_num == 1:
            L = 1.0
            B1 = 0.0
            B5 = 0.0
        else:
            # Get all bonds involving the atom
            bonds = atom.GetBonds()
            L_values = []
            B1_values = []
            B5_values = []

            for bond in bonds:
              # Get the indices of the two atoms in the bond
              bond_atom1 = bond.GetBeginAtomIdx()
              bond_atom2 = bond.GetEndAtomIdx()

              # Determine the other atom in the bond
              other_atom_idx = bond_atom2 if bond_atom1 == atom.GetIdx() else bond_atom1

              # Calculate Sterimol parameters for the bond
              sterimol_params, _ = self.local_sterimol(atom.GetIdx(), other_atom_idx)
              L_values.append(sterimol_params[0])
              B1_values.append(sterimol_params[1])
              B5_values.append(sterimol_params[2])

            # Calculate the average L, B1, and B5
            L = np.mean(L_values) if L_values else 0.0
            B1 = np.mean(B1_values) if B1_values else 0.0
            B5 = np.mean(B5_values) if B5_values else 0.0

        return [L, B1, B5]
    


import numpy as np
import trimesh
from trimesh.creation import ellipsoid

def compute_union_volume_surface(bond_parameters):
    """
    Computes the total volume and surface area of overlapping ellipsoids
    representing substituents around a central atom.

    Parameters:
    bond_parameters (list of dict): Each dict should have keys:
        - 'L': Length along the bond axis
        - 'B1': Minimum width perpendicular to the bond axis
        - 'B5': Maximum width perpendicular to the bond axis
        - 'direction': 3D unit vector indicating bond direction

    Returns:
    dict: Total volume and surface area of the union of ellipsoids.
    """
    ellipsoids = []

    for params in bond_parameters:
        L = params['L']
        B1 = params['B1']
        B5 = params['B5']
        direction = np.array(params['direction'])

        # Semi-axes
        a = L / 2.0
        b = B1 / 2.0
        c = B5 / 2.0

        # Create ellipsoid mesh
        ellipsoid_mesh = ellipsoid([a, b, c], subdivisions=3)

        # Align ellipsoid along the bond direction
        z_axis = np.array([0, 0, 1])
        rotation_matrix = trimesh.geometry.align_vectors(z_axis, direction)
        ellipsoid_mesh.apply_transform(rotation_matrix)

        # Translate ellipsoid to position along the bond
        ellipsoid_mesh.apply_translation(direction * a)

        ellipsoids.append(ellipsoid_mesh)

    # Perform boolean union of all ellipsoids
    combined = ellipsoids[0]
    for mesh in ellipsoids[1:]:
        combined = combined.union(mesh, engine='scad')  # Requires OpenSCAD installed

    return {
        'total_volume': combined.volume,
        'total_surface_area': combined.area
    }

        





from rdkit import Chem
from rdkit.Chem import AllChem, rdFreeSASA

def compute_atom_sasa_percentages(mol):
    # Ensure the molecule has 3D coordinates
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol)

    radii = rdFreeSASA.ClassifyAtoms(mol)
    sasa = rdFreeSASA.CalcSASA(mol, radii)

    atom_sasa = [mol.GetAtomWithIdx(i).GetProp('SASA') for i in range(mol.GetNumAtoms())]
    atom_sasa = list(map(float, atom_sasa))
    total_sasa = sum(atom_sasa)

    if total_sasa == 0:
        return [0.0 for _ in atom_sasa]

    return [100.0 * a / total_sasa for a in atom_sasa]

def compute_bond_sasa_percentages(mol):
    atom_sasa_percent = compute_atom_sasa_percentages(mol)
    bond_sasa_percent = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_exposure = (atom_sasa_percent[i] + atom_sasa_percent[j]) / 2.0
        bond_sasa_percent.append(bond_exposure)

    return bond_sasa_percent
