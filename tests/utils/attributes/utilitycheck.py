import sys
sys.path.append("/Users/svaddadi/Documents/GitHub/")
import EGAT
from EGAT import src
from EGAT.src.utils.descriptors.egat.encodings import Encodings 
from EGAT.src.utils.descriptors.egat.molmatdesc import MolMatDesc
from EGAT.src.utils.descriptors.egat.stereo import StereoChemistry
from EGAT.src.utils.descriptors.egat.radicals import RDKElectronInfo,YARPElectronInfo
from EGAT.src.utils.descriptors.egat.functional import FunctionalGroups
from EGAT.src.utils.descriptors.egat.reactive import Reactive
from rdkit import Chem 
from rich import inspect
from rich.console import Console

def get_dir_to_txt_file(object,filename):
    """
    Get the directory of the txt file
    """
    # Create a Console that writes to a file
    with open(filename, "w") as file:
        console = Console(file=file)
        inspect(object, console=console)


# Gives what Encodings are available
encodings = Encodings()
get_dir_to_txt_file(encodings,'encodings.txt')


# Give what descriptors are available
sample = 'CC1=CC=C(C=C1)C2=CC=CC=C2C3=CC=CC=C3C4=CC=CC=C4C5=CC=CC=C5'
descriptor = MolMatDesc(sample)
descriptor.run()
get_dir_to_txt_file(descriptor,'molmatdesc.txt')
stereo = StereoChemistry(descriptor)
stereo.run()
get_dir_to_txt_file(stereo,'stereo.txt')
rdkit = RDKElectronInfo(descriptor)
rdkit.run()
get_dir_to_txt_file(rdkit,'rdkitradicals.txt')
sample = 'CC1=CC=C(C=C1)C2=CC=CC=C2C3=CC=CC=C3C4=CC=CC=C4C5=CC=CC=C5'
sample = 'C1=COC=C1'
mol = Chem.MolFromSmiles(sample)
check = FunctionalGroups()
print(check.are_atoms_in_same_functional_group(mol, 0, 1))
get_dir_to_txt_file(check,'fg.txt')

rdkit = YARPElectronInfo(descriptor)
rdkit.run()
get_dir_to_txt_file(rdkit,'yarpradicals.txt')


sample = '[C:1][c:2]1[c:3][c:4][c:5][c:6][c:7]1'
reactant = Chem.MolFromSmiles(sample)
R_bond_mat = Chem.GetAdjacencyMatrix(reactant, useBO=True)
sample = '[CH3:1].[c:2]1[c:3][c:4][c:5][c:6][c:7]1'
product = Chem.MolFromSmiles(sample)
P_bond_mat = Chem.GetAdjacencyMatrix(product, useBO=True)
E = ['C', 'C', 'C', 'C', 'C', 'C', 'C']

reactive = Reactive(E, R_bond_mat, P_bond_mat)
reactive.ReactiveAtoms()
reactive.bnfn()

get_dir_to_txt_file(reactive,'reactive.txt')
get_dir_to_txt_file(reactive.bond,'reactivebond.txt')



