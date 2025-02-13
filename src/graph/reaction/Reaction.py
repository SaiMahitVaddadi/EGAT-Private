
from ..base.component import BaseReactionComponentFeaturizer
from ..base.information import ReactiveAtomInformation,ReactiveBondInformation,ReactiveAtomGeometryInformation,ReactiveBondGeometryInformation,GlobalReactionBondInformation,ReactiveAtomChangeInformation
from ..molecular.Molecule import MoleculeFeaturizer
from ..base.reaction import BaseReactionFeaturizer
from .helper import ReactionHelper
class ReactionComponentFeaturizer(BaseReactionComponentFeaturizer,MoleculeFeaturizer):
    def __init__(self, smiles, arguments):
        super(BaseReactionComponentFeaturizer, self).__init__(smiles, arguments)
        super(MoleculeFeaturizer, self).__init__(smiles, arguments)
    

class ReactionFeaturizer(BaseReactionFeaturizer,ReactionHelper,ReactiveAtomInformation,ReactiveBondInformation,ReactiveAtomChangeInformation):
    def __init__(self,reaction_smiles,arguments,denotation='>>'):
        super(BaseReactionFeaturizer, self).__init__(reaction_smiles, arguments,denotation,ReactionComponentFeaturizer)
    
    def AddOptionalInfo(self,mainattr,ind,edge=None,object=None):
        for attr in dir(mainattr):
            if callable(getattr(mainattr, attr)) and not attr.startswith("__"):
                bond_info_func = getattr(mainattr, attr)
                if edge is not None:
                    self.AddBondFunction(edge,ind,bond_info_func)
                else:
                    self.AddAtomFunction(ind,object,bond_info_func)

    def BondFeatureVector(self,ind):
        edge = sorted([self.reactant.edges_u[ind],self.reactant.edges_v[ind]])
        if self.params.oldbondencode: RBtype,PBtype = self.OldBondChangeInfo(edge)
        else: RBtype,PBtype = self.BondChangeInfo(edge)    
        self.reactant.bond_features[ind] += RBtype
        self.product.bond_features[ind] += PBtype
        self.AddOptionalInfo(ReactiveBondInformation,ind,edge)

        

    def AtomFeatureVector(self,ind,object):
        object.atom_features[ind] += self.DistanceFromReactingAtom(ind,object.gs)
        if self.params.addneighboringreactives: object.atom_features[ind] += self.NeighboringReactives(object.matrixdescriptors.adj_mat,ind)
        self.AddOptionalInfo(ReactiveAtomChangeInformation,ind,object)


    def GenerateBondFeatureVector(self):
        for ind in range(len(self.reactant.edges_u)):
            self.BondFeatureVector(ind)

            
    def GenerateAtomFeatureVector(self):
        for ind,atom_feature in enumerate(self.reactant.atom_features):
            self.AtomFeatureVector(ind,self.reactant)
        for ind,atom_feature in enumerate(self.product.atom_features):
            self.AtomFeatureVector(ind,self.product)

    def run(self):
        self.GenerateAtomFeatureVector()
        self.GenerateBondFeatureVector()        



        

