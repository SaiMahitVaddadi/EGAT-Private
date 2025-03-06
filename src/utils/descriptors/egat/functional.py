from rdkit import Chem



"""
FunctionalGroups class
======================

This class provides methods to work with various functional groups in organic chemistry.

Methods
-------

__init__(self)
    Initializes the FunctionalGroups class with a dictionary of functional groups and their corresponding SMARTS patterns.

get_functional_groups(self)
    Returns a list of all functional group names.

load_as_rdkit_mol(self, group_name)
    Loads a functional group as an RDKit molecule object from its SMARTS pattern.

get_substructure_match(self, mol, group_name)
    Finds the substructure match of a functional group in a given molecule.

are_atoms_in_same_functional_group(self, mol, atom_idx1, atom_idx2)
    Checks if two atoms in a molecule belong to the same functional group.

Attributes
----------

functional_groups : dict
    A dictionary where keys are functional group names and values are their corresponding SMARTS patterns. You can modify this dictionary to add or remove functional groups manually.




Example usage
-------------
fg = FunctionalGroups()
print(fg.get_functional_groups())
mol = fg.load_as_rdkit_mol('Alcohol')
print(Chem.MolToSmiles(mol))
"""


class FunctionalGroups:
    def __init__(self):
        self.functional_groups = {
            'Alcohol': 'CO',
            'Aldehyde': 'C=O',
            'Ketone': 'CC(=O)C',
            'Carboxylic Acid': 'C(=O)O',
            'Ester': 'CC(=O)OC',
            'Ether': 'COC',
            'Amine': 'CN',
            'Amide': 'C(=O)N',
            'Nitrile': 'C#N',
            'Thiol': 'CS',
            'Sulfide': 'CSC',
            'Disulfide': 'CSSC',
            'Halide': 'C[F,Cl,Br,I]',  # X = F, Cl, Br, I
            'Phenol': 'c1ccccc1O',
            'Aniline': 'c1ccccc1N',
            'Nitro': 'C[N+](=O)[O-]',
            'Sulfonic Acid': 'CS(=O)(=O)O',
            'Phosphate': 'CP(=O)(O)O',
            'Sulfoxide': 'CS(=O)C',
            'Sulfone': 'CS(=O)(=O)C',
            'Imine': 'C=NC',
            'Isocyanate': 'N=C=O',
            'Isothiocyanate': 'N=C=S',
            'Azide': 'N=[N+]=[N-]',
            'Diazo': 'N=N',
            'Hydrazone': 'NN',
            'Oxime': 'C=NO'
        }
        self.functional_groups.update({
            'Epoxide': 'C1OC1',
            'Aziridine': 'C1NC1',
            'Oxetane': 'C1CCO1',
            'Azetidine': 'C1CCN1',
            'Oxirane': 'C1CO1',
            'Azirine': 'C1CN1',
            'Oxazole': 'c1cocn1',
            'Isoxazole': 'c1nocn1',
            'Thiazole': 'c1cscn1',
            'Isoxazoline': 'c1noccn1',
            'Thiazoline': 'c1csccn1',
            'Pyridine': 'c1ccncc1',
            'Pyrrole': 'c1cccn1',
            'Furan': 'c1ccoc1',
            'Thiophene': 'c1ccsc1',
            'Imidazole': 'c1cncn1',
            'Pyrazole': 'c1cncn1',
            'Triazole': 'c1nncn1',
            'Tetrazole': 'c1nnnn1',
            'Pyrimidine': 'c1cncnc1',
            'Purine': 'c1ncncnc1',
            'Indole': 'c1ccc2c(c1)ccn2',
            'Quinoline': 'c1cc2cccnc2c1',
            'Isoquinoline': 'c1cc2cccnc2c1',
            'Benzimidazole': 'c1ccc2c(c1)ccn2',
            'Benzothiazole': 'c1ccc2c(c1)ccs2',
            'Benzoxazole': 'c1ccc2c(c1)cco2',
            'Benzofuran': 'c1ccc2c(c1)cco2',
            'Benzothiophene': 'c1ccc2c(c1)ccs2'
        })

        self.functional_groups.update({
            'Cyclopropane': 'C1CC1',
            'Cyclobutane': 'C1CCC1',
            'Cyclopentane': 'C1CCCC1',
            'Cyclohexane': 'C1CCCCC1',
            'Cycloheptane': 'C1CCCCCC1',
            'Cyclooctane': 'C1CCCCCCC1',
            'Cyclononane': 'C1CCCCCCCC1',
            'Cyclodecane': 'C1CCCCCCCCC1',
            'Cycloundecane': 'C1CCCCCCCCCC1',
            'Cyclododecane': 'C1CCCCCCCCCCC1',
            'Cyclotridecane': 'C1CCCCCCCCCCCC1',
            'Cyclotetradecane': 'C1CCCCCCCCCCCCC1',
            'Cyclopentadecane': 'C1CCCCCCCCCCCCCC1',
            'Cyclohexadecane': 'C1CCCCCCCCCCCCCCC1',
            'Cycloheptadecane': 'C1CCCCCCCCCCCCCCCC1',
            'Cyclooctadecane': 'C1CCCCCCCCCCCCCCCCC1',
            'Cyclononadecane': 'C1CCCCCCCCCCCCCCCCCC1',
            'Cycloeicosane': 'C1CCCCCCCCCCCCCCCCCCC1'
        })

    def get_functional_groups(self):
        return list(self.functional_groups.keys())

    def load_as_rdkit_mol(self, group_name):
        # Check if the provided group name exists in the functional groups dictionary
        if group_name in list(self.functional_groups.keys()):
            # Retrieve the SMARTS pattern for the given functional group name
            smarts = self.functional_groups[group_name]
            try:
                # Try to create an RDKit molecule object from the SMARTS pattern
                return Chem.MolFromSmarts(smarts)
            except:
                try:
                    # If the first attempt fails, try to convert the SMARTS pattern to a SMILES pattern
                    smarts = Chem.MolFromSmiles(smarts)
                    # Convert the SMILES pattern back to a SMARTS pattern
                    smarts = Chem.MolToSmarts(smarts)
                    # Try to create an RDKit molecule object from the new SMARTS pattern
                    return Chem.MolFromSmarts(smarts)
                except Exception as e:
                    # If both attempts fail, print an error message and return None
                    print(f"Error loading functional group '{group_name}': {e}")
                    return None
        else:
            # If the group name is not found in the dictionary, raise a ValueError
            raise ValueError(f"Functional group '{group_name}' not found.")
            
    def get_substructure_match(self, mol, group_name):
        # Load the functional group as an RDKit molecule object
        substructure = self.load_as_rdkit_mol(group_name)
        
        # If the substructure could not be created, raise an error
        if substructure is None:
            raise ValueError(f"Could not create substructure for '{group_name}'")
        
        # Find the substructure match in the given molecule
        match = mol.GetSubstructMatch(substructure)
        
        # If no match is found, return None for both atom and bond indices
        if not match:
            return None, None
        
        # Convert the match to a list of atom indices
        atom_indices = list(match)
        
        # Initialize an empty list to store bond indices
        bond_indices = []
        
        # Iterate over the bonds in the substructure
        for bond in substructure.GetBonds():
            # Get the start and end atoms of the bond in the match
            start_atom = match[bond.GetBeginAtomIdx()]
            end_atom = match[bond.GetEndAtomIdx()]
            
            # Get the bond index in the original molecule and add it to the bond_indices list
            bond_indices.append(mol.GetBondBetweenAtoms(start_atom, end_atom).GetIdx())
        
        # Return the atom indices and bond indices
        return atom_indices, bond_indices

    def are_atoms_in_same_functional_group(self, mol, atom_idx1, atom_idx2):
        # Iterate over all functional group names in the dictionary
        for group_name in self.functional_groups:
            try:
                # Try to get the substructure match for the current functional group in the given molecule
                atom_indices, _ = self.get_substructure_match(mol, group_name)
                
                # Check if both atom indices are in the list of matched atom indices
                if atom_indices and atom_idx1 in atom_indices and atom_idx2 in atom_indices:
                    # If both atoms are in the same functional group, return True
                    return True
            except Exception as e:
                # If an error occurs, continue to the next functional group
                continue
        
        # If no match is found for any functional group, return False
        return False

# Example usage:
# fg = FunctionalGroups()
# print(fg.get_functional_groups())
# mol = fg.load_as_rdkit_mol('Alcohol')
# print(Chem.MolToSmiles(mol))