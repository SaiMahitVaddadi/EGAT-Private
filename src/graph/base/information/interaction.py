
from ..base import BaseFeaturizer
from .kallisto import KallistoInformation
from rdkit import Chem

from math import floor

class JazzyCommands(BaseFeaturizer):
    def __init__(self, smiles, arguments):
        pass

    def _bondingpartners(self,idx):
        mol = self.matrixdescriptors.new_mol
        atom = mol.GetAtomWithIdx(idx)
        neighbors = [nbr.GetIdx() for nbr in atom.GetNeighbors()]
        return neighbors
    
    def _nearestneighbors(self,nbrs,center):
        mol = self.matrixdescriptors.new_mol
        nearest_neighbors = []
        for idx in nbrs:
            # Get the neighbors of the current atom
            atom = mol.GetAtomWithIdx(idx)
            neighbors = []
            for nbr in atom.GetNeighbors():
                if nbr.GetIdx() != center:
                    if nbr.GetIdx() not in nearest_neighbors:
                        if nbr.GetIdx() not in nbrs:
                            neighbors.append(nbr.GetIdx())
        return neighbors
        
    def _nearestnearestneighbors(self,nbrs,nearestnbrs,center):
        mol = self.matrixdescriptors.new_mol
        nearest_nearest_neighbors = []
        for idx in nearestnbrs:
            # Get the neighbors of the current atom
            atom = mol.GetAtomWithIdx(idx)
            neighbors = []
            for nbr in atom.GetNeighbors():
                if nbr.GetIdx() != center:
                    if nbr.GetIdx() not in nearest_nearest_neighbors:
                        if nbr.GetIdx() not in nbrs:
                            if nbr.GetIdx() not in nearestnbrs:
                                neighbors.append(nbr.GetIdx())
        return neighbors



class DeltaFunctions(JazzyCommands,KallistoInformation):
    def _init__(self):
        pass

    def _q_delta(self,q_alpha, q_beta, q_gamma, t):
        """Calculates q delta as per Equation 9 in Gerber's paper."""
        return t * sum(q_alpha) + (t**2) * sum(q_beta) + (t**3) * sum(q_gamma)

    def _charge(self,mol):
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
        charges = [
            float(atom.GetProp("_GasteigerCharge"))
            for atom in mol.GetAtoms()
        ]
        return charges
    
    def _eeq(self,id=1):
        self.ObtainAPC(id)

    def _q_nbrs(self,nbrs):
        """Calculates q alpha as per Equation 9 in Gerber's paper."""
        return sum([self.qs[nbr] for nbr in nbrs])
    
    def _donor_strength(self,atom_idx,nbrs,nn,nextnn,D=63.7,t=0.274,id=1):
        self._eeq(id)
        q = self.qs[atom_idx]
        q_alpha = self._q_nbrs(nbrs)
        q_beta = self._q_nbrs(nn)
        q_gamma = self._q_nbrs(nextnn)
        q_delta = self._q_delta(q_alpha, q_beta, q_gamma, t)
        return D*(q + q_delta)
    

    def _acceptor_strength(self,atom_idx,nbrs,nn,nextnn,A=6.1475,t=0.274,id=1):
        self._eeq(id)
        q = self.qs[atom_idx]
        q_alpha = self._q_nbrs(nbrs)
        q_beta = self._q_nbrs(nn)
        q_gamma = self._q_nbrs(nextnn)
        q_delta = self._q_delta(q_alpha, q_beta, q_gamma, t)
        return A*(q + q_delta)
    

    def getstrength(self,mol, atom_idx, d=63.7, t=0.274,id=1):
        nbrs = self._bondingpartners(atom_idx)
        nn = self._nearestneighbors(nbrs,atom_idx)
        nextnn = self._nearestnearestneighbors(nbrs,nn,atom_idx)
        if mol.GetAtomWithIdx(atom_idx).GetAtomicNum() == 1:
            """Calculates strength of the atom based on its type."""
            return self._donor_strength(atom_idx, nbrs, nn, nextnn, d, t,id)
        else:
            """Calculates strength of the atom based on its type."""
            return self._acceptor_strength(atom_idx, nbrs, nn, nextnn, d, t,id)


    def GetValence(self,atom):
        symbol = atom.GetSymbol()
        symbol = symbol.lower()
        self.valence = self.props.el_valence[symbol]
        self.alt_valence = self.props.el_alt_valence[symbol]
    
    def GetBonds(self,atom):
        bonds = 0
        num_aromatic_bonds = 0
        for neighbor in atom.GetNeighbors():
            bond = self.new_mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            if bond.GetIsAromatic():
                bonds += 2.0
                num_aromatic_bonds += 1
            else:
                bonds += bond.GetBondTypeAsDouble()
        if bond.GetIsAromatic():
            if num_aromatic_bonds > 1:
                bonds = bonds - (num_aromatic_bonds - 1)
        return bonds
    
    def EncodeLonePair(self,atom):
        self.GetValence(atom)
        lp = self.valence - self.GetBonds(atom) - atom.GetNumRadicalElectrons() - atom.GetFormalCharge()
        lp = self.ChangeForSpecialCases(lp,atom)
        return lp

    def ChangeForSpecialCases(self,lp,atom):
        if atom.GetSymbol() in ['P','S'] and lp < 0:
            lp = self.alt_valence - self.GetBonds(atom) - atom.GetNumRadicalElectrons() - atom.GetFormalCharge()
            if lp < 0: lp = 0
        else:
            if lp < 0 : lp = 0
        return lp

    def _get_lone_pairs(self, atom_idx):
        mol = self.matrixdescriptors.new_mol
        atom = mol.GetAtomWithIdx(atom_idx)
        lp = self.EncodeLonePair(atom)
        return floor(lp/2)
    
    def _get_hydrogens(self,atom_idx):
        mol = self.matrixdescriptors.new_mol
        atom = mol.GetAtomWithIdx(atom_idx)
        return atom.GetTotalNumHs()

    def local_g_polar(self,atom_idx,expd=.1,expa=.1,gd=.1,ga=.1,id=1):
        mol = self.matrixdescriptors.new_mol
        atoms = mol.GetAtoms()
        total_hs = sum([self._get_hydrogens(atom.GetIdx()) for atom in atoms if atoms.GetAtomicNum() != 1])
        lps = [self._get_lone_pairs(atom.GetIdx()) for atom in atoms if atoms.GetAtomicNum() != 1]
        total_lp = sum(lps)

        strength = self.getstrength(atom_idx,id=id)
        if mol.GetAtomWithIdx(atom_idx).GetAtomicNum() == 1:
            return gd * strength * total_hs ** expd
        else:
            if lps[atom_idx] > 0:
                return ga * strength * total_lp ** expa
            else:
                return 0
            
    def _interactive_contrib(self,atom_idx, expa=.1,id=1):
        strength = self.getstrength(atom_idx,id=id)
        lp = self._get_lone_pairs(atom_idx)
        return strength * lp ** expa

    def local_g_int(self,mol,atom_idx,gi=.1,k=1,id=1):
        nbrs = self._bondingpartners(atom_idx)
        nn = self._nearestneighbors(nbrs,atom_idx)
        nextnn = self._nearestnearestneighbors(nbrs,nn,atom_idx)
        a_a = self._interactive_contrib(atom_idx,id=id)
        alpha_int = sum([self._interactive_contrib(nbr,id=id) for nbr in nbrs])
        beta_int = sum([self._interactive_contrib(nbr,id=id) for nbr in nn])
        gamma_int = sum([self._interactive_contrib(nbr,id=id) for nbr in nextnn])

        return gi * a_a * (alpha_int + beta_int + k*gamma_int)

class InteractionInformation(DeltaFunctions):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
    
 
    def DeltaInteraction(self,ind,id=1):

        if self.matrixdescriptors.new_mol.GetAtoms()[ind].GetAtomicNum() == 1: 
            donororacceptor = [1,0,0]
        else:
            if self._get_lone_pairs(ind) > 0:
                donororacceptor = [0,1,0]
            else:
                donororacceptor = [0,0,1]
        
        strength = self.getstrength(ind,id=id)
        return strength,donororacceptor

        
    def FreeEnergies(self,ind,id=1):
        return [self.local_g_polar(ind,id=id),self.local_g_int(ind,id=1)]



  