from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

def get_tpsa_contributions(mol):
    total_tpsa, atom_contribs = rdMolDescriptors._CalcTPSAContribs(mol)
    return atom_contribs, total_tpsa




def GetAtomScoresFromSmarts(mol,smarts_logs):
    # Pre-compile SMARTS patterns
    smarts_patterns = [(Chem.MolFromSmarts(s), contrib) for s, contrib in smarts_logs.items()]

    atom_contributions = [0.0] * mol.GetNumAtoms()
    
    for smarts_mol, contrib in smarts_patterns:
        matches = mol.GetSubstructMatches(smarts_mol)
        for match in matches:
            for idx in match:
                atom_contributions[idx] += contrib
                    
    return atom_contributions


def compute_atomwise_logs(mol):

    smarts_logs = {
    "[NH0;X3;v3]": 0.71535,
    "[NH2;X3;v3]": 0.41056,
    "[nH0;X3]": 0.82535,
    "[OH0;X2;v2]": 0.31464,
    "[OH0;X1;v2]": 0.14787,
    "[OH1;X2;v2]": 0.62998,
    "[CH2;!R]": -0.35634,
    "[CH3;!R]": -0.33888,
    "[CH0;R]": -0.21912,
    "[CH2;R]": -0.23057,
    "[ch0]": -0.37570,
    "[ch1]": -0.22435,
    "F": -0.21728,
    "Cl": -0.49721,
    "Br": -0.57982,
    "I": -0.51547,}

    return GetAtomScoresFromSmarts(mol, smarts_logs)


from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit.Chem import rdMolDescriptors
def get_slogp_smr_contributions(mol):
    """
    Returns per-atom SlogP (L_i) and SMR (R_i) contributions for a molecule.
    
    Args:
        smiles (str): SMILES string of the molecule.
        include_hydrogens (bool): Whether to include explicit hydrogens.

    Returns:
        mol (rdkit.Chem.Mol): RDKit molecule object
        Li (list of float): SlogP atomic contributions
        Ri (list of float): SMR atomic contributions
    """

    logp_contribs = Crippen.rdMolDescriptors._CalcCrippenContribs(mol)
    Li = [x[0] for x in logp_contribs]
    Ri = [x[1] for x in logp_contribs]

    # Create binary one-hot vector for SlogP regions
    SlogP_region = []
    for li in Li:
        region = [0] * 10  # Initialize a binary vector of size 10
        if li <= -0.4:
            region[0] = 1  # SlogP_VSA0
        elif -0.4 < li <= -0.2:
            region[1] = 1  # SlogP_VSA1
        elif -0.2 < li <= 0:
            region[2] = 1  # SlogP_VSA2
        elif 0 < li <= 0.1:
            region[3] = 1  # SlogP_VSA3
        elif 0.1 < li <= 0.15:
            region[4] = 1  # SlogP_VSA4
        elif 0.15 < li <= 0.20:
            region[5] = 1  # SlogP_VSA5
        elif 0.20 < li <= 0.25:
            region[6] = 1  # SlogP_VSA6
        elif 0.25 < li <= 0.30:
            region[7] = 1  # SlogP_VSA7
        elif 0.30 < li <= 0.40:
            region[8] = 1  # SlogP_VSA8
        else:  # li > 0.40
            region[9] = 1  # SlogP_VSA9
        SlogP_region.append(region)

    SMR_region = []
    for ri in Ri:
        region = [0] * 8  # Initialize a binary vector of size 8
        if ri <= 0.11:
            region[0] = 1  # SMR_VSA0
        elif 0.11 < ri <= 0.26:
            region[1] = 1  # SMR_VSA1
        elif 0.26 < ri <= 0.35:
            region[2] = 1  # SMR_VSA2
        elif 0.35 < ri <= 0.39:
            region[3] = 1  # SMR_VSA3
        elif 0.39 < ri <= 0.44:
            region[4] = 1  # SMR_VSA4
        elif 0.44 < ri <= 0.485:
            region[5] = 1  # SMR_VSA5
        elif 0.485 < ri <= 0.56:
            region[6] = 1  # SMR_VSA6
        else:  # ri > 0.56
            region[7] = 1  # SMR_VSA7
        SMR_region.append(region)

    return Li, Ri, SlogP_region, SMR_region