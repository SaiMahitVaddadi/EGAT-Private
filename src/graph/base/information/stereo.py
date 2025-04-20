from ..base import BaseFeaturizer

class StereoInformation(BaseFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)

    def ChiralityCheck(self,ind):
        if not self.params.removechiralinfo:
            ###### CHECK IF IT IS IN A CHIRAL CENTER
            if ind in self.stereo.chiral_centers: chiral = self.properties.atom_chiral_encode[self.stereo.chiral_centers[ind]]
            else: chiral = [0,0,1]
            return chiral
        else:
            return []
        
    def BondConjugation(self,edge,bo):
        # Add Bond Conjugation info if not stated that you want to remove it from training. 
        if not self.params.removeconjinfo:
            if bo == 0:
                return [0]
            else:
                if tuple(edge) in self.stereo.conjugation and self.stereo.conjugation[tuple(edge)]: return [1]
                else: return [0]
        else:
            return []
    
    def BondStereochemistry(self,edge,bo):
        # Add Bond Conjugation info if not stated that you want to remove it from training. 
        if not self.params.removestereoinfo:
            if bo == 0:
                return [0,0,0]
            else:
                if tuple(edge) in self.stereo.bond_stereo and self.stereo.bond_stereo[tuple(edge)]: return self.properties.bond_stereo_encode[self.stereo.bond_stereo[tuple(edge)]]
                else: return [0,0,0]
        else:
            return []

        