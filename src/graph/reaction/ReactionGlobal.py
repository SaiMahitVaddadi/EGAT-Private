from ..base.component import BaseReactionComponentFeaturizer
from ..base.information import ReactiveAtomInformation,ReactiveBondInformation,GlobalReactionBondInformation,ReactiveAtomChangeInformation
from ..molecular.MoleculeGlobal import MoleculeFeaturizerwithPadding
from ..base.reaction import BaseReactionFeaturizer
from .helper import ReactionHelper
from tqdm import tqdm

class ReactionParams:
    oldbondencode: bool = False
    addneighboringreactives: bool = False

class ReactionComponentFeaturizer(BaseReactionComponentFeaturizer,MoleculeFeaturizerwithPadding):
    def __init__(self, smiles, arguments):
        super(BaseReactionComponentFeaturizer, self).__init__(smiles, arguments)
        super(MoleculeFeaturizerwithPadding, self).__init__(smiles, arguments)

class ReactionFeaturizerwithPadding(ReactiveAtomInformation,ReactiveBondInformation,ReactiveAtomChangeInformation):
    def __init__(self,reaction_smiles,arguments,denotation='>>'):
        self.reaction = reaction_smiles
        self.params = arguments
        self.basefeaturizer = MoleculeFeaturizerwithPadding
        self.denotation = denotation
        self.SetupFeaturizer()
        self.CreateMolecularVectors()
        self.GrabNetworkXFunctions()
        self.GrabReactiveAtomInformation()
        self.GetAllEdges()
    
    def CreateMolecularVectors(self):
        self.reactant.run()
        self.product.run()
        self.reqs_mat = [self.NeighboringReactives]

    def GetAllEdges(self):
        self.edges = max([len(self.reactant.all_edges),len(self.product.all_edges)])


    def BondFeatureVector(self,ind):
        funcs = [self.BondChangeInformation,self.DistanceFromReactingBond,self.BondOrderChange,self.BondNeighborhoodChange,self.ShortestPathChangeAcrossBond,self.BondParticipationDegree,self.BRICSBondRoleChange]
        params = ['removebondchangeinfo','adddisttoreactingbonds','getbondorderchange','getbondneighborhoodchange','getshortestpathchangeacrossbond','getbondparticipationdegree','getbricsbondrolechange']
        reactant_atom_features,reactant_bond_features,product_atom_features,product_bond_features,reactant_case,product_case = self.grabfeatures()
        for i, func in enumerate(tqdm(funcs, desc="Processing Bond Features")):
            self.obtainbondfcn(ind, reactant_bond_features, product_bond_features, reactant_case, product_case, Rmat=self.reactant.matrixdescriptors.adj_mat, Pmat=self.product.matrixdescriptors.adj_mat, removeparams=params[i], func=func)
      
    def AtomFeatureVector(self,ind):
        funcs = [
            self.DistanceFromReactingAtom,
            self.NeighboringReactives,
            self.ChangeInAtomicHybridization,
            self.DegreeCentralityOfChangingAtoms,
            self.ClosenessCentralityOfChangingAtoms,
            self.BetweennessCentralityOfChangingAtoms,
            self.EigenvectorCentralityOfChangingAtoms,
            self.KatzCentralityOfChangingAtoms,
            self.PageRankCentralityOfChangingAtoms,
            self.KCoreNumberOfChangingAtoms,
            self.HarmonicCentralityOfChangingAtoms,
            self.LocalBridgingOfChangingAtoms,
            self.TriangleCountOfChangingAtoms,
            self.LocalAtomFeaturesOfChangingAtoms,
            self.AtomicValencyChange,
            self.OxidationOrReduction,
            self.LocalBondOrderSumChange,
            self.AtomicNeighborhoodChangeRatio
        ]
        params = ['removebondchangeinfo','addneighboringreactives','gethybridizationchange',
                  'getclosenesscentrality','getdegreecentrality',
                'getbetweennesscentrality',
                'geteigenvectorcentrality',
                'getkatzcentrality',
                'getpagerankcentrality',
                'getkcorenumber',
                'getharmoniccentrality',
                'getlocalbridging',
                'gettrianglecount',
                'getlocalatomfeatures',
                'getvalencychange',
                'getoxidationreduction',
                'getbondordersumchange',
                'getneighborhoodchangeratio']
        reactant_atom_features,reactant_bond_features,product_atom_features,product_bond_features,reactant_case,product_case = self.grabfeatures()
        for i, func in enumerate(tqdm(funcs, desc="Processing Atom Features")):
            self.obtainatomfcn(ind, reactant_atom_features, product_atom_features, reactant_case, product_case, Rmat=self.reactant.matrixdescriptors.adj_mat, Pmat=self.product.matrixdescriptors.adj_mat, removeparams=params[i], func=func)

    def GenerateBondFeatureVector(self):
        for ind in range(self.edges):
            try:
                self.BondFeatureVector(ind)
            except:
                continue

            
    def GenerateAtomFeatureVector(self):
        for ind in range(len(self.reactant.matrixdescriptors.element)):
            self.AtomFeatureVector(ind)

    def run(self):
        self.GenerateAtomFeatureVector()
        self.GenerateBondFeatureVector()        



        