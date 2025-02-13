from rdkit import Chem

class FunctionalGroups:
    def __init__(self):
        self.functional_groups = {
            'Alcohol': 'C-OH',
            'Aldehyde': 'C=O',
            'Ketone': 'C(=O)C',
            'Carboxylic Acid': 'C(=O)OH',
            'Ester': 'C(=O)OC',
            'Ether': 'C-O-C',
            'Amine': 'C-N',
            'Amide': 'C(=O)N',
            'Nitrile': 'C#N',
            'Thiol': 'C-SH',
            'Sulfide': 'C-S-C',
            'Disulfide': 'C-S-S-C',
            'Halide': 'C-X',  # X = F, Cl, Br, I
            'Phenol': 'c1ccccc1O',
            'Aniline': 'c1ccccc1N',
            'Nitro': 'C-N(=O)=O',
            'Sulfonic Acid': 'C-S(=O)(=O)O',
            'Phosphate': 'C-O-P(=O)(O)O',
            'Sulfoxide': 'C-S(=O)-C',
            'Sulfone': 'C-S(=O)(=O)-C',
            'Imine': 'C=NH',
            'Isocyanate': 'N=C=O',
            'Isothiocyanate': 'N=C=S',
            'Azide': 'C-N=N=N',
            'Diazo': 'C=N=N',
            'Hydrazone': 'C=NN',
            'Oxime': 'C=NOH'
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

    def get_functional_groups(self):
        return list(self.functional_groups.keys())

    def load_as_rdkit_mol(self, group_name):
        if group_name in self.functional_groups:
            smarts = self.functional_groups[group_name]
            return Chem.MolFromSmarts(smarts)
        else:
            raise ValueError(f"Functional group '{group_name}' not found.")
    
    def get_substructure_match(self, mol, group_name):
        substructure = self.load_as_rdkit_mol(group_name)
        if substructure is None:
            raise ValueError(f"Could not create substructure for '{group_name}'")
        match = mol.GetSubstructMatch(substructure)
        if not match:
            return None
        atom_indices = list(match)
        bond_indices = []
        for bond in substructure.GetBonds():
            start_atom = match[bond.GetBeginAtomIdx()]
            end_atom = match[bond.GetEndAtomIdx()]
            bond_indices.append(mol.GetBondBetweenAtoms(start_atom, end_atom).GetIdx())
        return atom_indices, bond_indices

    def are_atoms_in_same_functional_group(self, mol, atom_idx1, atom_idx2):
        for group_name in self.functional_groups:
            try:
                atom_indices, _ = self.get_substructure_match(mol, group_name)
                if atom_indices and atom_idx1 in atom_indices and atom_idx2 in atom_indices:
                    return True
            except ValueError:
                continue
        return False

# Example usage:
# fg = FunctionalGroups()
# print(fg.get_functional_groups())
# mol = fg.load_as_rdkit_mol('Alcohol')
# print(Chem.MolToSmiles(mol))