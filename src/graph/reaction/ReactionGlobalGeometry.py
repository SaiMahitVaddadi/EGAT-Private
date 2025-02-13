
from ..base.component import BaseReactionComponentFeaturizer
from ..base.information import ReactiveAtomInformation,ReactiveBondInformation,ReactiveAtomGeometryInformation,ReactiveBondGeometryInformation,GlobalReactionBondInformation,ReactiveAtomChangeInformation
from ..molecular.MoleculeGlobalGeometry import MoleculeFeaturizerwithPaddingandGeometry
from ..base.reaction import BaseReactionFeaturizer
from .helper import ReactionHelper
from .Reaction import ReactionFeaturizer

class ReactionComponentFeaturizer(BaseReactionComponentFeaturizer,MoleculeFeaturizerwithPaddingandGeometry):
    def __init__(self, smiles, arguments):
        super(BaseReactionComponentFeaturizer, self).__init__(smiles, arguments)
        super(MoleculeFeaturizerwithPaddingandGeometry, self).__init__(smiles, arguments)
    
class ReactionFeaturizerwithPaddingandGeometry(BaseReactionFeaturizer,ReactionFeaturizer,ReactionHelper,ReactiveAtomInformation,ReactiveBondInformation,ReactiveAtomChangeInformation,GlobalReactionBondInformation,ReactiveAtomGeometryInformation,ReactiveBondGeometryInformation):
    def __init__(self,reaction_smiles,arguments,denotation='>>'):
        super(BaseReactionFeaturizer, self).__init__(reaction_smiles, arguments,denotation,ReactionComponentFeaturizer)
    
    def BondFeatureVector(self,ind):
        edge = sorted([self.reactant.edges_u[ind],self.reactant.edges_v[ind]])
        if self.params.oldbondencode: RBtype,PBtype = self.OldBondChangeInfo(edge)
        else: RBtype,PBtype = self.BondChangeInfo(edge)    
        self.reactant.bond_features[ind] += RBtype
        self.product.bond_features[ind] += PBtype
        self.AddOptionalInfo(ReactiveBondInformation,ind,edge)
        self.AddOptionalInfo(GlobalReactionBondInformation,ind,edge)
        self.AddOptionalInfo(ReactiveBondGeometryInformation,ind,edge)
        
    def AtomFeatureVector(self,ind,object):
        object.atom_features[ind] += self.DistanceFromReactingAtom(ind,object.gs)
        if self.params.addneighboringreactives: object.atom_features[ind] += self.NeighboringReactives(object.matrixdescriptors.adj_mat,ind)
        self.AddOptionalInfo(ReactiveAtomChangeInformation,ind,object)
        self.AddOptionalInfo(ReactiveAtomGeometryInformation,ind,object)
        
    
