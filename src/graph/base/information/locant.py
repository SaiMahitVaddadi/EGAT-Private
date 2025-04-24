from ..base import BaseFeaturizer
from ....utils.descriptors.egat.functional import FunctionalGroups

class LocantInformation(BaseFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)

    def GrabFunctionalGroups(self):
        self.fgclass = FunctionalGroups()
        self.fgs = self.fgclass.functional_groups
        # Filter functional groups to include only those with 'O' or 'N' and no numbers
        self.fgs = {
            key: value
            for key, value in self.fgclass.functional_groups.items()
            if ('O' in key or 'N' in key) and not any(char.isdigit() for char in key)
        }
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

    def GrabLocantsForFunctionalGroup(self,fg):
        atoms,bonds = self.fgclass.get_substructure_match(self.matrixdescriptors.new_mol,fg)
        if atoms is None:
            return dict()
        else:
            locants = dict()
            for atom in atoms:
                nbrs = self._bondingpartners(atom)
                nearestnbrs = self._nearestneighbors(nbrs,atom)
                nearestnearestnbrs = self._nearestnearestneighbors(nbrs,nearestnbrs,atom)
            
            locants['alpha'] = [n for n in nbrs if n not in atoms]
            locants['beta'] = [n for n in nearestnbrs if n not in atoms]
            locants['gamma'] = [n for n in nearestnearestnbrs if n not in atoms]
            return locants
    

    def GrabLocantsForFunctionalGroups(self):
        self.GrabFunctionalGroups()
        self.locantlist = []
        for fg in self.fgs:
            locants = self.GrabLocantsForFunctionalGroup(fg)
            if locants:
                self.locantlist.append(locants)
    

    def _add_to_dict(self,alist,key1):
        for val in alist:
            if val not in self.locantcount[key1]:
                self.locantcount[key1][val] = 1
            else:
                self.locantcount[key1][val] += 1
    
    def GrabLocantCount(self):
        self.GrabLocantsForFunctionalGroups()

        self.locantcount = dict()
        self.locantcount['alpha'] = dict()
        self.locantcount['beta'] = dict()
        self.locantcount['gamma'] = dict()

        for locants in self.locantlist:
            alphas = locants['alpha']
            betas = locants['beta']
            gammas = locants['gamma']

            self._add_to_dict(alphas,'alpha')
            self._add_to_dict(gammas,'gamma')
            self._add_to_dict(betas,'beta')
    
    def LocantCountForAtom(self,ind):
        if self.params.getlocantcount == 'all': 
            return [self.locantcount['alpha'][ind],self.locantcount['beta'][ind],self.locantcount['gamma'][ind]]
        elif self.params.getlocantcount == 'carbononly':
            if self.matrixdescriptors.new_mol.GetAtomWithIdx(ind).GetSymbol() == 'C':
                return [self.locantcount['alpha'][ind],self.locantcount['beta'][ind],self.locantcount['gamma'][ind]]
            else:
                return [0,0,0]
        else:
            return []





