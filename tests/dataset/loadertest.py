import sys
def trace(frame, event, arg):
    print("%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno))
    return trace
sys.path.append("/Users/svaddadi/Documents/GitHub/")
import EGAT
from EGAT import src
from EGAT.src.dataset.base.dataset_setups import DatasetParams
from EGAT.src.params.graph import GraphParams
from EGAT.src.loader.Loader import EGATDataLoader
import torch
import gc
gc.disable()
torch.set_num_threads(1)
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

params.conformer.generator = 'RDKit'
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
params.getMoRSEweights= 'mass'  # Weights for MoRSE descriptors
params.getMoRSEbins = 10  # Number of bins for MoRSE descriptors
params.getclusteringcoeff = True
params.getclosenesscentrality = True
params.getdegreecentrality = True
params.getgraphdensity = True
params.getavgdegneighbors = True
params.getnodeclustering = True
params.getassortativity = True
params.getspectralradius = True
params.getdegreeentropy = True
params.getlocalbertzct = True
params.getcoulomb = True
params.getpagerank = True
params.getglobal = True
params.getshortestpath = 'all'
params.bias_walk = 'random'
params.num_walks = 10
params.getuniquewalks = True
params.getrandomwalk = True
params.getshortestpathcount = True
params.getcommonneighbors = 'all'
params.getglobaljaccard = True
params.getglobaladamicadar = True
params.getprefattachment = True
params.getkatz = True
params.getEigenvectorCentrality = True
params.getBetweennessCentralityCorrelation = True
params.getMinCutValue = True
params.getMaximumFlow = True
params.getLaplacianEigenvectorSimilarity = True
params.getFiedlerVectorSimilarity = True
params.getBetweennessCentralityOfPathways = True
params.getRingsInSharedPath = True
params.getfirstpassagetime = True
params.geteffectiveresistance = True
params.getSameAromaticSequence = True
params.getTopoOverlap = True
params.getEdgeClustering = True
params.getFormanCurve = True
params.global_fingerprint_type = 'Morgan'
params.global_similarity_metric = 'Tanimoto'
params.getLocalAtomicEnvironmentSimilarity = True
params.getsamefg = True
params.ct_method = 'spectral'
params.addhbonds = True
params.addcho = True
params.adddihydrogenbonds = True
params.addcationpi = True
params.addpipistack = True
params.addhalogenbonds = True
params.addmetallophilic = True
params.addelectrostatic = True
params.check_hbond = True
params.wt_adj_mat_by = None  # Options: 'bond_order', 'atomic_mass', 'valence', 'hybridization', 'coulomb', 'all'
params.rw_weight_by = None  # Options: 'atomic_mass', 'bond_order', 'hyb', 'valence', 'coulomb', 'all'
params.sp_box_size = None  # For periodic boundary conditions
params.hop_radius = 1  # For k-hop subgraph extraction
params.getrandomwalkcommutetime  = True

# Create new DatasetParams
dataset_params = DatasetParams(data_path='/Users/svaddadi/Documents/GitHub/EGAT/tests/dataset/data/',
                               rootfile='/Users/svaddadi/Documents/GitHub/EGAT/tests/dataset/example.csv',
                               smiles='smiles',
                               target='target')
dataset_params.graph = 'molecular'
dataset_params.addons = None
dataset_params.denoteby = None
dataset_params.dimension = '2d'
dataset_params.mode = "dgl"
dataset_params.jepa = False
dataset_params.globalmode = False
dataset_params.class_choice = None
dataset_params.test_class_choice = None
dataset_params.notcase = False
dataset_params.test_notcase = False
dataset_params.test_only = False
dataset_params.dataset = 'onthefly'
dataset_params.fingerprint = False
dataset_params.llm = False
dataset_params.imblearn = False 
dataset_params.shuffle_loader = False
dataset_params.randomize= True
dataset_params.batch_size = 5
dataset_params.shuffle = False
dataset_params.num_workers = 0
dataset_params.drop_last = False
dataset_params.combine_loader = None
# Merge DatasetParams and GraphParams
for attr, value in vars(dataset_params).items():
    setattr(params, attr, value)
params.target = ['target','additional_1']
params.additional = ['additional_1']
params.smiles = ['smiles','solvant']
from rich import inspect

#inspect(dataset, methods=True, all=True)


# Initialize the data loader
data_loader = EGATDataLoader(params)

# Call the data loader to prepare datasets and dataloaders
data_loader()

# Access the dataloaders
for split, loader in data_loader.egatdataloader.items():
    print(f"DataLoader for {split}:")
    print(loader,type(loader))
    try:
        for id,rtypes,Rgs,smiles,targets in loader:
            print(targets)
    except:
        for batch in loader:
            print('Batch: ',batch)
            #print(batch['Rgraphs'])
            #print(batch['targets'])
            #print(batch['smiles'])
            #print(batch['types'])
            #print(batch['names'])
            #print(batch['additionals'])
            #print(batch['Radd'])
            #print(batch['Rgraphs'])
            #print(batch['targets'])
            #print(batch['smiles'])
            #print(batch['types'])
            #print(batch['names'])
            break



# Add additional parameters to params
params.seed = None
params.gpu = 0
params.save_path = "./"
params.trainer = "EGAT"
params.startpoint = None
params.base_model = None
params.ablation_EGAT_model = None
params.ablation_NN_model = None
params.parallel = False
params.finetuner = None
params.defaults = None
params.learning_rate = 0.001
params.weight_decay = 0.0001
params.batch_size = 32
params.num_epochs = 1
params.setup = 'cpu'  # Options: 'cuda', 'mps', 'rocm', 'cpu'
params.parallel = False  # Whether to use multiple devices if available
params.loss = 'L1'
params.metric = 'L1'
params.optimizer = 'adam'
params.lr = .0001
params.weight_decay = 0.0
params.momentum = 0.0
params.scheduler = 'cosine'
params.learning_rate_min = 0.0
params.momentum_orig = 0.1
params.lr_decay = 0.1
params.step_size = 10
params.epochs = 100
params.gamma = 0.1
params.base_lr = 0.001
params.max_lr = 0.01
params.step_size_up = 2000
params.steps_per_epoch = 100
params.lr_lambda = None
params.warmup_steps = 0
params.weightsandbiases = False
params.wandb_project = None
params.wandb_name = None


params.hidden_dim = 64
params.num_heads = 8
params.Resid = False
params.ResidBias = False
params.SA = False
params.MessagePassing = True
params.egatlayers = 2
params.getattentionmaps = None  # Options: 'norm', 'mean', 'median', or None
params.AggregateReaction = None  # Options: 'Concat', 'Mixing', etc.
params.Aggregate = 'Concat'  # Options: 'Concat', etc.
params.MixingLayer = False  # Options: 'Bilinear', 'Linear', True, False
params.pooling = 'Sum'  # Options: 'WtSum', 'LearnedAttn', 'CalcedAttn', etc.
params.norm = 1.0  # Normalization factor
params.architecture = '3MLP'  # '1MLP', '3MLP', 'custom', 'subnet'
params.model_type = 'direct'  # 'Hr', 'Hr2', or other types
params.activation = None  # Activation function for MLPs
params.dropout = 0.0  # Dropout rate for custom MLP
params.cascading = False  # Whether cascading is enabled
params.smax = False  # Whether softmax is applied
params.endconvolution = False  # Whether end convolution is applied
params.Bias = True  # Whether bias is used in layers

from EGAT.src.main.ml.setup import MLSetup

# Initialize the ML setup
ml_setup = MLSetup(params)
ml_setup.Load()