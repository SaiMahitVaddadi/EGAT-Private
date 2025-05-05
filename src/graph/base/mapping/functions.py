from ..base import BaseFeaturizer
from ....utils.descriptors.egat.encodings import Encodings
from ....utils.descriptors.egat.reactive import Reactive
from rdkit import Chem
import warnings
from dataclasses import dataclass
from typing import Any, Optional
from rxnmapper import RXNMapper


class AtomMapping:
    def __init__(self,arguments):
        self.params = arguments
    
    def MapviaRXNMapper(self,r,p):
        rxn_mapper = RXNMapper()
        rxn = [r + '>>'+p]
        results = rxn_mapper.get_attention_guided_atom_maps(rxn)
        # Get Mapped Reaction
        self.mapper_condfidence = results[0]['confidence'] 
        results = results[0]['mapped_rxn'].split('>>')
        rsmi = results[0]
        psmi = results[1]
        if self.params.totalmapping:
            rsmi = self.AddHMapping(rsmi)
            psmi = self.AddHMapping(psmi)
        return rsmi,psmi

    def MapviaLocalMapper(self,r,p):
        pass
    
    def MapviaAAMUtils(self,r,p):
        pass
    
    def MapviaRDKit(self):
        pass

    def MapviaOB3D(self):
        pass

    def MapviaOB2D(self):
        pass

    def AddHMapping(self,smi):
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        for atom in mol.GetAtoms():
            if atom.GetSymbol() in ['H','Cl','F','I','Br']:
                atom.SetAtomMapNum(atom.GetIdx() + 1)  # Increment the atom mapping label for H atoms
        return Chem.MolToSmiles(mol)

    def StripAtomMapping(self, smi):
        molecule = Chem.MolFromSmiles(smi)
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(molecule)

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
        if not self.IsAtomMapped(self.reactant) or not self.IsAtomMapped(self.product):
            r = self.StripAtomMapping(self.reactant)
            p = self.StripAtomMapping(self.product)

            mapfunction = getattr(self,f'Mapvia{self.params.mappingfunction}',None)
            self.reactant,self.product = mapfunction(r,p)
    