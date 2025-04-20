from ..base import BaseFeaturizer

class RDInformation(BaseFeaturizer):    
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
    
    def SpiroCheck(self,ind):
        if self.params.getspiro:
            if ind in self.matrixdescriptors.spiro:
                return [0,1]
            else:
                return [1,0]
        else:
            return []

    def BridgeHeadCheck(self,ind):
        if self.params.getbridgehead:
            if ind in self.matrixdescriptors.bridgehead:
                return [0,1]
            else:
                return [1,0]
        else:
            return []
    
    def BondRotation(self,edge,bo):
        # Add Bond Conjugation info if not stated that you want to remove it from training. 
        if self.params.getrotatablebonds:
            if bo == 0:
                return [0,0]
            else:
                if tuple(edge) in self.matrixdescriptors.rotationalbond:
                    return self.properties.bond_rotat_encode['TRUE']
                else:
                    return self.properties.bond_rotat_encode['FALSE']
        else:
            return []
        