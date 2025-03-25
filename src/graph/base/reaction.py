from .base import BaseFeaturizer
from ...utils.descriptors.egat.encodings import Encodings
from ...utils.descriptors.egat.reactive import Reactive
from rdkit import Chem
import warnings
from dataclasses import dataclass
from typing import Any, Optional
from rxnmapper import RXNMapper
from localmapper import localmapper
from aamutils.aam_expand import extend_aam_from_rsmi

@dataclass
class ReactionBaseParams:
    mappingfunction: str = 'RXNMapper'
    totalmapping: Optional[bool] = True
class BaseReactionFeaturizer:
    def __init__(self,reaction_smiles,arguments,denotation='>>',reactionfeaturizer=BaseFeaturizer):
        self.reaction = reaction_smiles
        self.params = arguments
        self.SplitandCheckForMapping(denotation)
        self.reactant = reactionfeaturizer(self.reactant,arguments)
        self.product = reactionfeaturizer(self.product,arguments)
       
        self.elementcheck = self.CheckElementConsistency()
        if self.elementcheck: self.ReactiveBondInformation()


    def IsAtomMapped(self,smi):
        molecule = Chem.MolFromSmiles(smi)
        for i,atom in enumerate(molecule.GetAtoms()):
            if atom.HasProp('molAtomMapNumber'):
                if i > 0 and atom.GetAtomMapNum() == 0:
                    return False
            else:
                return False
        return True

    def SplitandCheckForMapping(self,denotation):
        self.reactant = self.reaction.split(denotation)[0]
        self.product = self.reaction.split(denotation)[1]
        print(f"Reactant: {self.reactant}, Product: {self.product}")

        if not self.IsAtomMapped(self.reactant) or not self.IsAtomMapped(self.product):
            r = self.StripAtomMapping(self.reactant)
            p = self.StripAtomMapping(self.product)

            mapfunction = getattr(self,f'Mapvia{self.params.mappingfunction}',None)
            self.reactant,self.product = mapfunction(r,p)
    

    def MapviaRDKit(self):
        pass

    def MapviaOB3D(self):
        pass

    def MapviaOB2D(self):
        pass

    def MapviaRXNMapper(self,r,p):
        rxn_mapper = RXNMapper()
        rxn = [r + '>>'+p]
        results = rxn_mapper.get_attention_guided_atom_maps(rxn)
        # Get Mapped Reaction
        self.mapper_condfidence = results[0]['confidence'] 
        results = results[0]['mapped_rxn'].split('>>')
        rsmi = results[0]
        psmi = results[1]
        print(f"Reactant: {rsmi}, Product: {psmi}")
        if self.params.totalmapping:
            rsmi = self.AddHMapping(rsmi)
            psmi = self.AddHMapping(psmi)
        return rsmi,psmi

    def MapviaLocalMapper(self,r,p):
        mapper = localmapper()
        rxn = r + '>>'+p
        result = mapper.get_atom_map(rxn,return_dict=True)
        results = result['mapped_rxn'].split('>>')
        self.template = result['template']
        self.mapper_condfidence = result['confidence']
    
    

    def MapviaAAMUtils(self,r,p):
        result_smiles = extend_aam_from_rsmi(r + '>>'+p)

    def AddHMapping(self,smi):
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        for atom in mol.GetAtoms():
            if atom.GetSymbol() in ['H','Cl','F','I','Br']:
                atom.SetAtomMapNum(atom.GetIdx() + 1)  # Increment the atom mapping label for H atoms
        return Chem.MolToSmiles(mol)

    def StripAtomMapping(self, smi):
        molecule = Chem.MolFromSmiles(smi)
        print(molecule,smi)
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(molecule)

    def SetFeaturesToNone(self):
        self.reactant.atom_features = None
        self.product.atom_features = None
        self.reactant.bond_features = None
        self.product.bond_features = None
            
    def CheckElementConsistency(self):
        if self.reactant.matrixdescriptors.element != self.product.matrixdescriptors.element and len(self.reactant.matrixdescriptors.element) == len(self.product.matrixdescriptors.element):
            warnings.warn("Element inconsistency between reactant and product, but the same # of atoms are there.")
            self.SetFeaturesToNone()
            return False
        elif self.reactant.matrixdescriptors.element != self.product.matrixdescriptors.element and len(self.reactant.matrixdescriptors.element) != len(self.product.matrixdescriptors.element):
            warnings.warn("Element inconsistency between reactant and product.")
            self.SetFeaturesToNone()
            return False
        else:
            return False
        
    def ReactiveBondInformation(self):
        self.reactionpropeties = Reactive(self.reactant.matrixdescriptors.element,self.reactant.matrixdescriptors.bond_mat,self.product.matrixdescriptors.bond_mat)
    