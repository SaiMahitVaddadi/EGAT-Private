from rdkit import Chem

def match_all_smarts(mol):
    smarts_table = [
        (1,  "[OH1][*]",                     "sOH"),
        (2,  "O=[*]",                         "dO"),
        (3,  "[OH0]([*])[*]",                "ssO"),
        (4,  "[o]",                           "aaO"),
        (5,  "[NH2][*]",                      "sNH2"),
        (6,  "[NH1]=[*]",                     "dNH"),
        (7,  "[NH1]([*])[*]",                 "ssNH"),
        (8,  "[nH1]",                         "aaNH"),
        (9,  "N#[*]",                         "tN"),
        (10, "[ND2](=[*])[*]",               "dsN"),
        (11, "[nH0]",                         "aaN"),
        (12, "N([*])([*])[*]",                "sssN"),
        (13, "N(=[*])(=[*])[*]",              "ddsN"),
        (14, "[N;+]([*])([*])([*])[*]",       "ssssN+"),
        (15, "[SH1][*]",                      "sSH"),
        (16, "S=[*]",                         "dS"),
        (17, "[SX2]([*])[*]",                 "ssS"),
        (18, "[s]",                           "aaS"),
        (19, "S(=[*])(=[*])([*])[*]",         "ddssS"),
        (20, "[F][*]",                        "sF"),
        (21, "[Cl][*]",                       "sCl"),
        (22, "[Br][*]",                       "sBr"),
        (23, "[I][*]",                        "sI"),
        (24, "[CH3][*]",                      "sCH3"),
        (25, "[CH2]([*])[*]",                 "ssCH2"),
        (26, "[CH2]=[*]",                     "dCH2"),
        (27, "[CH1]([*])([*])[*]",            "sssCH1"),
        (28, "[CH1](=[*])[*]",                "dsCH1"),
        (29, "[CH1]#[*]",                     "tCH"),
        (30, "[cH]",                          "aaCH"),
        (31, "[cH0]",                         "aasC"),
        (32, "C(=[*])=[*]",                   "ddC"),
        (33, "C(#[*])[*]",                    "tsC"),
        (34, "C(=[*])([*])[*]",               "dssC"),
        (35, "C([*])([*])([*])[*]",           "ssssC"),
    ]

    matches = {}
    for row_no, smarts, label in smarts_table:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            matches[label] = []
            continue

        atom_matches = mol.GetSubstructMatches(pattern)
        # Flatten matches to get all individual atom indices
        atom_indices = sorted(set(i for match in atom_matches for i in match))
        matches[label] = atom_indices

    # Create a dictionary to store the vector for each atom
    atom_estate_vectors = {atom.GetIdx(): [0] * len(smarts_table) for atom in mol.GetAtoms()}

    # Populate the vectors with the number of hits for each estate
    for estate_index, (_, _, label) in enumerate(smarts_table):
        for atom_idx in matches[label]:
            atom_estate_vectors[atom_idx][estate_index] += 1

    # Add the atom_estate_vectors to the matches dictionary
    return atom_estate_vectors
