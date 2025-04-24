import sys
sys.path.append("/Users/svaddadi/Documents/GitHub/")
import EGAT
from EGAT import src
from EGAT.src.graph.molecular.Molecule import MoleculeFeaturizer
from EGAT.src.graph.base.base import BaseFeaturizer,BaseReactionParams
from EGAT.src.graph.base.component import BaseReactionComponentFeaturizer
from EGAT.src.graph.molecular.MoleculeGeometry import MoleculeFeaturizerwithGeometry
from EGAT.src.params.graph import GraphParams

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


params = BaseReactionParams()
params.acidbase = 'BL'
params.getradical = 'RDKit'
params.stereo_full = True
smiles = "CC(=O)C1=CC=C(C=C1)C(=O)O"

graph = BaseFeaturizer(smiles,params)
get_dir_to_txt_file(graph,'Base.txt')

graph= BaseReactionComponentFeaturizer(smiles,params)
get_dir_to_txt_file(graph,'BaseReaction.txt')

params = GraphParams()
# This is the way to set the parameters if you want explicit Acid-Base Information
params.acidbase = 'Lewis'
params.getacidbaseinfo = True
# This is the way to get radicals
# Note: For YARP,  YARP can give multiple smiles based on the resonance structure. The way we handle this is by obtaining x different graphs for the same y.
# With that, we would need to change the dataloader to handle such cases as it also happens with the conformational sampling. 
params.getradical = 'YARP'
# This is the way to get spiro and brdgehead information. 
params.getspiro = True
params.getbridgehead = True
# This is the way to get fused ring information.
params.getfusedinformation = True
# This is the way to get rotatable bond information.
params.getrotatablebonds = True
# This is the way to get gasteiger charges.
params.charge = 'Gasteiger'
# This is the way to get BRICS Decomposition locations. 
params.getbrics = True
params.checkbricsbond = True 
# This is the way to get Electronegativity locations. 
params.getelectronegativity = True
# This is the way to get the bond dipoles.
params.getdipole = True

graph = MoleculeFeaturizer('CCCCC',params)
graph.run()
get_dir_to_txt_file(graph,'Molecule.txt')


#This is how you set conformers
params.conformer.generator = 'RDKit'

params.getsterichindrance = True
params.getasa = True
params.getgaussiancurvature = True
params.getmolecularshapeindex = True
params.getdistancetoconvexhull = True
params.getbondlength = True
params.getbondangle = True
params.getdihedral = 'main'
graph = MoleculeFeaturizerwithGeometry('CCCCC',params)
graph.run()
get_dir_to_txt_file(graph,'MoleculewGeom.txt')