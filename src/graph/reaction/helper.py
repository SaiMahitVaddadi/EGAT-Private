from ..base.component import BaseReactionComponentFeaturizer
from ..base.information import ReactiveAtomInformation,ReactiveBondInformation,GlobalReactionBondInformation,ReactiveAtomChangeInformation
from ..base.reaction import BaseReactionFeaturizer

class ReactionHelperParams:
    oldbondencode: bool = False

class ReactionHelper(ReactiveAtomInformation,ReactiveBondInformation,GlobalReactionBondInformation,ReactiveAtomChangeInformation):
    def __init__(self,reaction_smiles,arguments,component_class,denotation='>>'):
        self.reaction = reaction_smiles
        self.params = arguments
        self.basefeaturizer = component_class
        self.denotation = denotation
        self.SetupFeaturizer()

    def _obtainatomfcn(self,ind,mat=None,removeparams=None,func=None):
        if not getattr(self.params,removeparams):
            if self.params.getradical == 'YARP':
                if self.reactant.electroninfo.yarpecule.atom_mats > 1:
                    for yarpid in self.reactant.atom_features_dict.keys():
                        for obj in [self.reactant,self.product]:
                            obj.atom_features_dict[yarpid][ind] += func(ind,mat)
                else:
                    for obj in [self.reactant,self.product]:
                            obj.atom_features[ind] += func(ind,mat)
            else:
                for obj in [self.reactant,self.product]:
                     obj.atom_features[ind] += func(ind,mat)



    
    def BondChangeInfo(self,edge,ind,id=0):
        if not self.params.removebondchangeinfo:
            if self.params.oldbondencode: RBtype,PBtype = self.OldBondChangeInfo(edge)
            else: RBtype,PBtype = self.BondChangeInfo(edge)
            if self.params.getradical == 'YARP': 
                if self.reactant.electroninfo.yarpecule.bond_mats > 1: 
                    for yarpid in self.reactant.bond_features_dict.keys():
                        self.reactant.bond_features_dict[yarpid][ind] += RBtype
                        self.product.bond_features_dict[yarpid][ind] += PBtype
                else:
                    self.reactant.bond_features[ind] += RBtype
                    self.product.bond_features[ind] += PBtype
            else:   
                self.reactant.bond_features[ind] += RBtype
                self.product.bond_features[ind] += PBtype

    def ReactingAtomInfo(self,ind,gs):
        self._obtainatomfcn(ind,mat=gs,removeparams='removebondchangeinfo',func=self.DistanceFromReactingAtom)

    def NeighboringReactives(self,adj_mat,ind):
        self._obtainatomfcn(ind,mat=adj_mat,removeparams='addneighboringreactives',func=self.NeighboringReactives)
        