from ..base import BaseFeaturizer
from rdkit import Chem

from mordred import _atomic_property
from dataclasses import dataclass


@dataclass
class ElectronParams:
    getradical: str = 'RDKit'
    removehybridinfo: bool = False
    useFullHyb: bool = False
    getpielectrons: bool = False
    getsigmaelectrons: bool = False
    getintrinsicstate: bool = False
    getetabeta: bool = False
    getionizationpotential: bool = False
    getcoreelectrons: bool = False
    getramificationnumber: bool = False


class ElectronInformation(BaseFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
    
    
    def RadicalCheck(self,ind,yarpid=0):
        if self.params.getradical == 'RDKit':
            return [self.electroninfo.rads[ind],self.electroninfo.lps[ind]]
        elif self.params.getradical == 'YARP':
            return [self.electroninfo.rads[yarpid][ind],self.electroninfo.lps[yarpid][ind]]
        else:
            return []
    
    def HybridizationCheck(self,ind,ind_=0):
        ###### GET HYBRIDIZATION
        if not self.params.removehybridinfo:
            if self.matrixdescriptors.element[ind_] == 'H':
                if not self.params.useFullHyb: hybrid = [0,0,0,1]
                else: hybrid = [0,0,0,1,0,0,0,0,0]

            else:
                hybrid = self.properties.atom_hybrid_encode[self.stereo.Hybridization[ind]]
            return hybrid
        else:
            return []
        
    def PiElectrons(self,ind):
        if self.params.getpielectrons:
            atom = self.matrixdescriptors.new_mol.GetAtomWithIdx(ind)
            pibonds = 0 
            for b in atom.GetBonds():
                if b.GetBondType() == Chem.BondType.DOUBLE:
                    pibonds += 1
                elif b.GetBondType() == Chem.BondType.TRIPLE:
                    pibonds += 2
                if b.GetIsAromatic():
                    pibonds += 1
            return [pibonds]
        else:
            return []
    
    def SigmaElectrons(self,ind):
        if self.params.getsigmaelectrons:
            atom = self.matrixdescriptors.new_mol.GetAtomWithIdx(ind)
            sigmabonds = 0 
            for b in atom.GetBonds():
                if b.GetBondType() == Chem.BondType.SINGLE:
                    sigmabonds += 1
                elif b.GetBondType() == Chem.BondType.DOUBLE:
                    sigmabonds += 1
                elif b.GetBondType() == Chem.BondType.TRIPLE:
                    sigmabonds += 1
                if b.GetIsAromatic():
                    sigmabonds += 1
            return [sigmabonds]
        else:
            return []

    def IntrinsicState(self,ind):
        if self.params.getintrinsicstate:
            atom = self.matrixdescriptors.new_mol.GetAtomWithIdx(ind)
            return [_atomic_property.get_intrinsic_state(atom)]
        else:
            return []

    def EtaBeta(self,ind):
        if self.params.getetabeta: 
            atom = self.matrixdescriptors.new_mol.GetAtomWithIdx(ind)
            return [_atomic_property.get_eta_beta_delta(atom),
                    _atomic_property.get_eta_beta_non_sigma(atom),
                    _atomic_property.get_eta_beta_sigma(atom),
                    _atomic_property.get_eta_gamma(atom),
                    _atomic_property.get_eta_epsilon(atom)]
        else:
            return []

    def EtaBond(self, edge):
        if self.params.getetabeta:
            atom1 = edge[0]
            atom2 = edge[1]
            bond = self.matrixdescriptors.new_mol.GetBondBetweenAtoms(atom1, atom2)
            if bond:
                return [_atomic_property.get_eta_nonsigma_contribute(bond)]
            else:
                return []
        else:
            return []

    def IonizationPotential(self,ind):
        if self.params.getionizationpotential:
            atom = self.matrixdescriptors.new_mol.GetAtomWithIdx(ind)

            return [_atomic_property.get_ionization_potential(atom)]
        else:
            return []    
        
    def CoreElectrons(self,ind):
        if self.params.getcoreelectrons:
            atom = self.matrixdescriptors.new_mol.GetAtomWithIdx(ind)
            return [_atomic_property.get_core_count(atom)]
        else:
            return []
    
    def RamificationNumber(self,ind):
        if self.params.getramificationnumber:
            atom = self.matrixdescriptors.new_mol.GetAtomWithIdx(ind)
            pibonds = 0 
            sigmabonds = 0 
            for b in atom.GetBonds():
                if b.GetBondType() == Chem.BondType.DOUBLE:
                    pibonds += 1
                elif b.GetBondType() == Chem.BondType.TRIPLE:
                    pibonds += 2
                if b.GetIsAromatic():
                    pibonds += 1

                if b.GetBondType() == Chem.BondType.SINGLE:
                    sigmabonds += 1
            
            ve = _atomic_property.get_valence_electrons(atom)

            lp = (ve - pibonds - sigmabonds - atom.GetFormalCharge() - atom.GetNumRadicalElectrons())/2

            return [lp + pibonds,lp + pibonds + sigmabonds]
        else:
            return []
        
    def SurroundingIPFeatures(self, ind):
        if self.params.getionizationpotentialpooled:
            atom = self.matrixdescriptors.new_mol.GetAtomWithIdx(ind)
            neighbor_ips = []
            for neighbor in atom.GetNeighbors():
                neighbor_ip = _atomic_property.get_ionization_potential(neighbor)
                neighbor_ips.append(neighbor_ip)
            
            if neighbor_ips:
                avg_ip = sum(neighbor_ips) / len(neighbor_ips)
                median_ip = sorted(neighbor_ips)[len(neighbor_ips) // 2]
                min_ip = min(neighbor_ips)
                max_ip = max(neighbor_ips)
                std_ip = (sum((x - avg_ip) ** 2 for x in neighbor_ips) / len(neighbor_ips)) ** 0.5
                return [avg_ip, median_ip, min_ip, max_ip, std_ip]
            else:
                return [0, 0, 0, 0, 0]
        else:
            return []


class ChargeInformation(BaseFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
        
    
    def ElectronegativityCheck(self,ind):
        if self.params.getelectronegativity:
            return [self.matrixdescriptors.electronegativity[ind]]
        else:
            return []
        
    def FormalCharge(self,ind):
        if not self.params.removeformalchargeinfo:
            fc = self.matrixdescriptors.fc[ind]
            return [fc]
        else:
            return []

    def ChargeCheck(self,ind):
        if self.params.charge == 'Gasteiger':
            return [float(self.matrixdescriptors.gasteiger_charges[ind])]
        elif self.params.charge == 'psi4':
            return []
        elif self.params.charge == 'pyscf':
            return []
        elif self.params.charge == 'gaussian':
            return []
        elif self.params.charge == 'orca':
            return []
        elif self.params.charge == 'ase':
            return []
        elif self.params.charge == 'gemnet':
            return []
        elif self.params.charge == 'chemprop':
            return []
        elif self.params.charge == 'unimol':
            return []
        else:
            return []
        
