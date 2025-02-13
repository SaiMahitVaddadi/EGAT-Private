from ..base.component import BaseReactionComponentFeaturizer
from ..base.information import ReactiveAtomInformation,ReactiveBondInformation,ReactiveAtomGeometryInformation,ReactiveBondGeometryInformation
from ..molecular.MoleculeGlobalGeometry import MoleculeFeaturizerwithPaddingandGeometry
from ..base.reaction import BaseReactionFeaturizer



class ReactionHelper(BaseReactionFeaturizer,ReactiveAtomInformation,ReactiveBondInformation,ReactiveAtomGeometryInformation,ReactiveBondGeometryInformation,GlobalReactionBondInformation,ReactiveAtomChangeInformation):
    def __init__(self,reaction_smiles,arguments,component_class,denotation='>>',):
        super(BaseReactionFeaturizer, self).__init__(reaction_smiles, arguments,denotation,component_class)
    
    def BondChangeInfo(self,edge,ind):
        if self.params.oldbondencode: RBtype,PBtype = self.OldBondChangeInfo(edge)
        else: RBtype,PBtype = self.BondChangeInfo(edge)    
        self.reactant.bond_features[ind] += RBtype
        self.product.bond_features[ind] += PBtype

    def AddBondFunction(self,edge,ind,fcn):
        if getattr(self,f'add{fcn.lower()}'):
            func = getattr(self,fcn)
            self.reactant.bond_features[ind] += func(edge)
            self.product.bond_features[ind] += func(edge)
    
    def AddAtomFunction(self,ind,fcn,object):
        if getattr(self,f'add{fcn.lower()}'): 
            func = getattr(self,fcn)
            object.atom_features[ind] += func(ind)