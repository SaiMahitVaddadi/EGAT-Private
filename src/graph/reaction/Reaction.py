
from ..base.component import BaseReactionComponentFeaturizer
from ..base.information import ReactiveAtomInformation,ReactiveBondInformation
from ..molecular.Molecule import MoleculeFeaturizer
from ..base.reaction import BaseReactionFeaturizer

class ReactionComponentFeaturizer(BaseReactionComponentFeaturizer,MoleculeFeaturizer):
    def __init__(self, smiles, arguments):
        super(BaseReactionComponentFeaturizer, self).__init__(smiles, arguments)
        super(MoleculeFeaturizer, self).__init__(smiles, arguments)
    


class ReactionFeaturizer(BaseReactionFeaturizer,ReactiveAtomInformation,ReactiveBondInformation):
    def __init__(self,reaction_smiles,arguments,denotation='>>'):
        super(BaseReactionFeaturizer, self).__init__(reaction_smiles, arguments,denotation,ReactionComponentFeaturizer)
    
    def BondFeatureVector(self,ind):
        edge = sorted([self.reactant.edges_u[ind],self.reactant.edges_v[ind]])
        if self.params.oldbondencode: RBtype,PBtype = self.OldBondChangeInfo(edge)
        else: RBtype,PBtype = self.BondChangeInfo(edge)    
        self.reactant.bond_features[ind] += RBtype
        self.product.bond_features[ind] += PBtype
        if self.params.addbonddistancechange:
            self.reactant.bond_features[ind] += self.BondDistanceChange(edge)
            self.product.bond_features[ind] += self.BondDistanceChange(edge)
        

    def AtomFeatureVector(self,ind,object):
        object.atom_features[ind] += self.DistanceFromReactingAtom(ind,object.gs)
        if self.params.addneighboringreactives: object.atom_features[ind] += self.NeighboringReactives(object.matrixdescriptors.adj_mat,ind)

    def GenerateBondFeatureVector(self):
        for ind in range(len(self.reactant.edges_u)):
            edge = sorted([self.reactant.edges_u[ind],self.reactant.edges_v[ind]])
            if self.params.oldbondencode: RBtype,PBtype = self.OldBondChangeInfo(edge)
            else: RBtype,PBtype = self.BondChangeInfo(edge)    
            self.reactant.bond_features[ind] += RBtype
            self.product.bond_features[ind] += PBtype
            if self.params.addbonddistancechange:
                self.reactant.bond_features[ind] += self.BondDistanceChange(edge)
                self.product.bond_features[ind] += self.BondDistanceChange(edge)

            
    def GenerateAtomFeatureVector(self):
        
        for ind,atom_feature in enumerate(self.reactant.atom_features):
            self.AtomFeatureVector(ind,self.reactant)
    
        for ind,atom_feature in enumerate(self.product.atom_features):
            self.AtomFeatureVector(ind,self.product)

    def run(self):
        self.GenerateAtomFeatureVector()
        self.GenerateBondFeatureVector()        



        

