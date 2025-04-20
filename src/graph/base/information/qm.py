# Atom and Bond QM Properties
'''
Atom properties:

- Energy (G,U,S,E)
- Dipole moment (D)
- Polarizability (P)
- Quadrupole moment (Q)
- Magnetic susceptibility (M)
- Hyperpolarizability (H)
- NMR chemical shift (N)
- NMR shielding (S)
- NMR coupling constant (C)
- Electron affinity (A)
- pKa

Bond properties:
- Fukui 
- NBO
- hirshfeld
- NPA
- NLMO

Methods: 
- pySCF
- ORCA
- Gaussian
- GAMESS
- Q-Chem
- Molpro
- NWChem
- Psi4
'''





class QMecule:
    def __init__(self, name, charge, multiplicity, atoms, bonds):
        self.name = name
        self.charge = charge
        self.multiplicity = multiplicity
        self.atoms = atoms
        self.bonds = bonds

    def get_atom_properties(self):
        atom_properties = {}
        for atom in self.atoms:
            atom_properties[atom] = {
                'energy': None,
                'dipole_moment': None,
                'polarizability': None,
                'quadrupole_moment': None,
                'magnetic_susceptibility': None,
                'hyperpolarizability': None,
                'nmr_chemical_shift': None,
                'nmr_shielding': None,
                'nmr_coupling_constant': None,
                'electron_affinity': None,
                'pKa': None
            }
        return atom_properties