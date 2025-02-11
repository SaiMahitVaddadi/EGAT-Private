from .base import BaseFeaturizer
from ...utils.descriptors.egat.encodings import Encodings
from ...utils.descriptors.egat.reactive import Reactive
from rdkit import Chem

class BaseReactionFeaturizer:
    def __init__(self,reaction_smiles,arguments,denotation='>>',reactionfeaturizer=BaseFeaturizer):
        self.reaction = reaction_smiles
        self.SplitandCheckForMapping(denotation)
        self.reactant = reactionfeaturizer(self.reactant,arguments)
        self.product = reactionfeaturizer(self.product,arguments)
        self.reactant.run()
        self.product.run()
        self.params = arguments
        self.elementcheck = self.CheckElementConsistency()
        if self.elementcheck: self.ReactiveBondInformation()

    def SplitandCheckForMapping(self,denotation):
        self.reactant = self.reaction.split(denotation)[0]
        self.product = self.reaction.split(denotation)[1]

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
        results = results[0]['mapped_rxn'].split('>>')
        rsmi = results[0]
        psmi = results[1]

        #Add Hydrogens and Map Them
        self.mapper_condfidence = results[0]['confidence'] 
        if self.params.rxnmapper.totalmapping:
            rmol = self.AddHMapping(rsmi)
            pmol = self.AddHMapping(psmi)
            rsmi = Chem.MolToSmiles(rmol)
            psmi = Chem.MolToSmiles(pmol)
        return rsmi,psmi

    def AddHMapping(smi):
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
    