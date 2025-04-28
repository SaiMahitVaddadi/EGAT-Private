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

'''
params = BaseReactionParasms()
params.acidbase = 'BL'
params.getradical = 'RDKit'
params.stereo_full = True
smiles = "CC(=O)C1=CC=C(C=C1)C(=O)O"

graph = BaseFeaturizer(smiles,params)
get_dir_to_txt_file(graph,'Base.txt')

graph= BaseReactionComponentFeaturizer(smiles,params)
get_dir_to_txt_file(graph,'BaseReaction.txt')
'''
params = GraphParams()
# This is the way to set the parameters if you want explicit Acid-Base Information
params.acidbase = 'Lewis'
params.getacidbaseinfo = True
# This is the way to get radicals
# Note: For YARP,  YARP can give multiple smiles based on the resonance structure. The way we handle this is by obtaining x different graphs for the same y.
# With that, we would need to change the dataloader to handle such cases as it also happens with the conformational sampling. 
params.getradical = 'RDKit'
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
# This is the way to get the sigma and pi electrons.
params.getpielectrons = True
params.getsigmaelectrons = True
# This is the way to get the intrinsic state.
params.getintrinsicstate = True
# This is the way to get the eta and beta values.
params.getetabeta = True
# This is the way to get the ionization potential.
params.getionizationpotential = True
params.getionizationpotentialpooled = True
# This get core electrons
params.getcoreelectrons = True
# this gets the ramification number. 
params.getramificationnumber = True
params.getjazzydelta=  True
params.getjazzyfe = True
params.getlocantcount=True
params.getpolarity = True
params.threshold = 10.0
params.covalentnumber = "exp"
params.ps_size = (1.0, 1.5)
params.kallisto_thresholdCN = 10.0
params.kallisto_thresholdBond = 0.5
params.vdw_scale = 1.0
params.vdwtype = "truhlar"
params.getCN = True
params.getPS = True
params.getAPC = True
params.getvdwkallisto = True
params.getAP = True
params.useabc = 'reg'
params.useabs = 'gg'
params.getbarysz = True
params.getats = 'ATS'
params.getdetour = True
params.getburdenmat = 'reg'
params.burdenweight = 'reg'
params.getpropsrelativetocarbon = False #Needs Work
params.getvertexdistancedegree = True
params.getbalabanbondfactor = True
params.getsuperdentic = True
params.geteccentricity = True
params.getschultz = True
params.getgutman = True
params.getxui = True
params.gethorary = True
params.getmohar = 'laplacian'
params.gethp = True
params.getcorecount = True
params.getvem = True
params.getetacomposite = True
params.getetapsi = True
params.getgravity = True
params.getzagreb = 2
params.getharmonic = True
params.getsombor = True
params.getrandic = True
params.getnirmala = True
params.getsoss = True
params.getaugmentedgraphattributes = True
params.gethyperbolic = True
params.getaugzagreb = True
params.getklein = True
params.gethyperwiener = True
params.getwiener = True
params.gethdsa = True
params.getets = True
params.getinformationcontent = 0
params.getweightedinformationcontent = 0
params.getmoleculardistanceedge = True
params.gettopocharge = 3
params.getSMR = ['Li', 'logPregion','Ri']
params.getMoRSE = 'cos'
params.getchi = 3
params.getedgewiener = True
params.getvertexadjacency = True
params.getTPSA = True
params.getASA = True
params.LogS = True
params.getEstate = 'all'
params.getchivalence = 3
params.MDEvalences = [1, 1]
params.MDEreference = 'C'
params.mohar = 'unnormalized'
params.getvewi = True
params.getvewibyorder = 3
params.getedgewienerbyorder = 3
params.IC_weight = 'mass'
params.getresonance = True
graph = MoleculeFeaturizer('CCCCC',params)
graph.run()
get_dir_to_txt_file(graph,'Molecule.txt')

params.conformer.generator = 1
params.conformer.nconfs = 1
params.getsterimol = True
params.local_cutoff = 4.0
params.use_vdw = True
params.removecoordinationinfo = True
params.getdistancetocenterofmass = True
params.getsterichindrance = True
params.getasa = True
params.getgaussiancurvature = True
params.getmolecularshapeindex = True
params.getdistancetoconvexhull = True
params.getvdw = True
params.getvdwstrain = True
params.getburiedvolume = True
params.getbondlength = True
params.getatomdistance = True
params.getbondangle = None
params.getdihedral = None
params.getbondmidpoint = True
params.getmomentdescriptors = ['geometric','sterimol']
params.getsolidanglecoverage = True 
graph = MoleculeFeaturizerwithGeometry('CCCCC',params)
graph.run()
get_dir_to_txt_file(graph,'MoleculeGeometry.txt')
'''
getsterimol: bool = False
    local_cutoff: float = 4.0
    use_vdw: bool = False
    removecoordinationinfo: bool = False
    getdistancetocenterofmass: bool = False
    getsterichindrance: bool = False
    getasa: bool = False
    getgaussiancurvature: bool = False
    getmolecularshapeindex: bool = False
    getdistancetoconvexhull: bool = False
    getvdw: bool = False
    getvdwstrain: bool = False
    getburiedvolume: bool = False
    getbondlength: bool = False
    getatomdistance: bool = False
    getbondangle: str = None
    getdihedral: str = None
    getbondmidpoint: bool = False
'''

