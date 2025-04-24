from ..base import BaseFeaturizer
from rdkit import Chem

from mordred import _atomic_property


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
        if self.getetabeta: 
            atom = self.matrixdescriptors.new_mol.GetAtomWithIdx(ind)
            return [_atomic_property.get_eta_beta_delta(atom),
                    _atomic_property.get_eta_beta_non_sigma(atom),
                    _atomic_property.get_eta_beta_sigma(atom),
                    _atomic_property.get_eta_gamma(atom),
                    _atomic_property.get_eta_epsilon(atom),
                    _atomic_property.get_eta_nonsigma_contribute(atom)]
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
            return [_atomic_property.get_core_electrons(atom)]
        else:
            return []
    
    def RamificationNumber(self):
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
        


def get_core_count(atom):
    Z = atom.GetAtomicNum()
    if Z == 1:
        return 0.0

    Zv = _table.GetNOuterElecs(Z)
    PN = period[Z]

    return (Z - Zv) / (Zv * (PN - 1))


def get_eta_epsilon(atom):
    Zv = _table.GetNOuterElecs(atom.GetAtomicNum())
    return 0.3 * Zv - get_core_count(atom)


def get_eta_beta_sigma(atom):
    e = get_eta_epsilon(atom)
    return sum(
        0.5 if abs(get_eta_epsilon(a) - e) <= 0.3 else 0.75
        for a in atom.GetNeighbors()
        if a.GetAtomicNum() != 1
    )


def get_eta_nonsigma_contribute(bond):
    if bond.GetBondType() is Chem.BondType.SINGLE:
        return 0.0

    f = 1.0
    if bond.GetBondTypeAsDouble() == Chem.BondType.TRIPLE:
        f = 2.0

    a = bond.GetBeginAtom()
    b = bond.GetEndAtom()

    dEps = abs(get_eta_epsilon(a) - get_eta_epsilon(b))

    if bond.GetIsAromatic():
        y = 2.0
    elif dEps > 0.3:
        y = 1.5
    else:
        y = 1.0

    return y * f


def get_eta_beta_delta(atom):
    if (
        atom.GetIsAromatic()
        or atom.IsInRing()
        or _table.GetNOuterElecs(atom.GetAtomicNum()) - atom.GetTotalValence() <= 0
    ):
        return 0.0

    for b in atom.GetNeighbors():
        if b.GetIsAromatic():
            return 0.5

    return 0.0



def get_eta_beta_non_sigma(atom):
    return sum(
        get_eta_nonsigma_contribute(b)
        for b in atom.GetBonds()
        if get_other_atom(b, atom).GetAtomicNum() != 1
    )


def get_eta_gamma(atom):
    beta = (
        get_eta_beta_sigma(atom)
        + get_eta_beta_non_sigma(atom)
        + get_eta_beta_delta(atom)
    )
    if beta == 0:
        return np.nan

    return get_core_count(atom) / beta



'''