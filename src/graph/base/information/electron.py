from ..base import BaseFeaturizer





class ElectronInformation(BaseFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
    
    
    def RadicalCheck(self,ind):
        if self.params.getradical:
            return [self.electroninfo.rads[ind],self.electroninfo.lps[ind]]
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