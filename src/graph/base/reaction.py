from .base import BaseFeaturizer
from ...utils.descriptors.egat.encodings import Encodings
from ...utils.descriptors.egat.reactive import Reactive
from .mapping.functions import AtomMapping
from rdkit import Chem
import warnings
from dataclasses import dataclass
from typing import Any, Optional
from rxnmapper import RXNMapper
@dataclass
class ReactionBaseParams:
    mappingfunction: str = 'RXNMapper'
    totalmapping: Optional[bool] = True

class BaseReactionFeaturizer(AtomMapping):
    def __init__(self,reaction_smiles,arguments,denotation='>>',reactionfeaturizer=BaseFeaturizer):
        self.reaction = reaction_smiles
        self.params = arguments
        self.basefeaturizer = reactionfeaturizer
        self.denotation = denotation
        self.SetupFeaturizer()
    
    def SetupFeaturizer(self):
        self.SplitandCheckForMapping(self.denotation)
        self.reactant = self.basefeaturizer(self.reactant,self.params)
        self.product = self.basefeaturizer(self.product,self.params)
       
        self.elementcheck = self.CheckElementConsistency()
        if self.elementcheck: self.ReactiveBondInformation()

    def SetFeaturesToNone(self):
        self.reactant.atom_features = None
        self.product.atom_features = None
        self.reactant.bond_features = None
        self.product.bond_features = None
            
    def CheckElementConsistency(self):
        if self.reactant.matrixdescriptors.element != self.product.matrixdescriptors.element and len(self.reactant.matrixdescriptors.element) == len(self.product.matrixdescriptors.element):
            warnings.warn("Element inconsistency between reactant and product, but the same # of atoms are there.")
            print("Element inconsistency between reactant and product, but the same # of atoms are there.")
            self.SetFeaturesToNone()
            return False
        elif self.reactant.matrixdescriptors.element != self.product.matrixdescriptors.element and len(self.reactant.matrixdescriptors.element) != len(self.product.matrixdescriptors.element):
            warnings.warn("Element inconsistency between reactant and product.")
            print("Element inconsistency between reactant and product.")
            self.SetFeaturesToNone()
            return False
        else:
            return True
        
    def ReactiveBondInformation(self):
        self.reactionpropeties = Reactive(self.reactant.matrixdescriptors.element,self.reactant.matrixdescriptors.bond_mat,self.product.matrixdescriptors.bond_mat)
    
    def determinecasetouse(self):
        if self.params.getradical == 'YARP':
            if not self.params.conformer.nconfs > 1:
                if self.reactant.electroninfo.yarpecule.atom_mats > 1 and self.product.electroninfo.yarpecule.bond_mats > 1:
                    return 1,1
                elif self.reactant.electroninfo.yarpecule.atom_mats > 1:
                    return 1,0
                elif self.product.electroninfo.yarpecule.bond_mats > 1:
                    return 0,1
                else:
                    return 0,0
            else:
                if self.reactant.electroninfo.yarpecule.atom_mats > 1 and self.product.electroninfo.yarpecule.bond_mats > 1:
                    return 2,2
                elif self.reactant.electroninfo.yarpecule.atom_mats > 1:
                    return 2,1
                elif self.product.electroninfo.yarpecule.bond_mats > 1:
                    return 1,2
                else:
                    return 1,1
        else:
            if self.params.conformer.nconfs > 1:
                return 2,2
            else:
                return 0,0
        
    def determinevectortouse(self,feat,case=0):
        if case == 0:
            return feat.atom_features,feat.bond_features
        elif case == 1:
            return feat.atom_features_dict,feat.bond_features_dict
        elif case == 2:
            return feat.atom_features_confs,feat.bond_features_confs

    def grabfeatures(self):
        reactant_case,product_case = self.determinecasetouse()
        reactant_atom_features,reactant_bond_features = self.determinevectortouse(self.reactant,reactant_case)
        product_atom_features,product_bond_features = self.determinevectortouse(self.product,product_case)
        return reactant_atom_features,reactant_bond_features,product_atom_features,product_bond_features,reactant_case,product_case

    def runfunction(self,func,ind,mat=None,product=False):
        if product:
            if func in self.reqs_mat:
                return func(ind,mat)
            else:
                return func(ind)
        else:
            if func in self.reqs_mat:
                return func(ind,mat,product)
            else:
                return func(ind,product)

    def iterateoverkeys(self,featdict,func,ind,mat=None,product=False):
        for key in featdict.keys():
            if func in self.reqs_mat:
                featdict[key][ind] += self.runfunction(func,ind,mat,product)

    def iterateoverindices(self,feat,func,ind,mat=None,product=False):    
        feat[ind] += self.runfunction(func,ind,mat,product)


    def iterateoverkeysbonds(self,featdict,func,ind,edge,mat=None,product=False):
        for key in featdict.keys():
            if func in self.reqs_mat:
                featdict[key][ind] += self.runfunction(func,edge,mat,product)

    def iterateoverindicesbonds(self,feat,func,ind,edge,mat=None,product=False):    
        feat[ind] += self.runfunction(func,edge,mat,product)

    def obtainatomfcn(self,ind,reactant_atom_features,product_atom_features,reactant_case,product_case,Rmat=None,Pmat=None,removeparams=None,func=None):
        if not getattr(self.params,removeparams):
            if reactant_case == 0: self.iterateoverindices(reactant_atom_features,func,ind,Rmat)
            else: self.iterateoverkeys(reactant_atom_features,func,ind,Rmat)
            if product_case == 0: self.iterateoverindices(product_atom_features,func,ind,Pmat,product=True)
            else: self.iterateoverkeys(product_atom_features,func,ind,Pmat,product=True)

    def edgefcn(self,ind,globalfeats=False,product=False):
        if globalfeats and product: edge = self.product.all_edges[ind]
        if not globalfeats and product: edge = sorted([self.product.edges_u[ind],self.product.edges_v[ind]])
        if not globalfeats and not product: edge = sorted([self.reactant.edges_u[ind],self.reactant.edges_v[ind]])
        if globalfeats and not product: edge = self.reactant.all_edges[ind]
        return edge

    def obtainbondfcn(self,ind,reactant_atom_features,product_atom_features,reactant_case,product_case,Rmat=None,Pmat=None,removeparams=None,func=None,globalfeats=None):
        if not getattr(self.params,removeparams):
            if reactant_case == 0: 
                edge = self.edgefcn(ind,globalfeats=globalfeats)
                self.iterateoverindicesbonds(reactant_atom_features,func,ind,edge,Rmat)
            else: 
                edge = self.edgefcn(ind,globalfeats=globalfeats)
                self.iterateoverkeysbonds(reactant_atom_features,func,ind,edge,Rmat)
            if product_case == 0: 
                edge = self.edgefcn(ind,globalfeats=globalfeats,product=True)
                self.iterateoverindicesbonds(product_atom_features,func,ind,edge,Pmat,product=True)
            else:
                edge = self.edgefcn(ind,globalfeats=globalfeats,product=True)
                self.iterateoverkeysbonds(product_atom_features,func,ind,edge,Pmat,product=True)

    
    