

class TautEnum:
    def __init__(self):
        pass

    def enumerate_tautomers(self):

        enumerator = rdMolStandardize.TautomerEnumerator()
        tautomers = enumerator.Enumerate(self.mol)

        self.tautomers = [Chem.MolToSmiles(tautomer) for tautomer in tautomers]
        return self.tautomers
        
    
    def enumerate_v1_tautomers(self):
        enumerator = rdMolStandardize.GetV1TautomerEnumerator()
        tautomers = enumerator.Enumerate(self.mol)

        self.v1_tautomers = [Chem.MolToSmiles(tautomer) for tautomer in tautomers]
        return self.tautomers
    
    def enumerate_molvs_tautomers(self):
        enumerator = TautomerEnumerator()
        tautomers = enumerator.enumerate(self.mol)

        self.molvs_tautomers = [Chem.MolToSmiles(tautomer) for tautomer in tautomers]
        return self.molvs_tautomers
