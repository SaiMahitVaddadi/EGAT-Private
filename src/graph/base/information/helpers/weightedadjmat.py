from .pdmetrics import PathDistanceMetrics
import numpy as np

class WeightedAdjacencyMatrix(PathDistanceMetrics):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)

    def getmasses(self):
        try:
            masses = [atom.GetMass() for atom in self.matrixdescriptors.element]
            return masses
        except Exception as e:
            print(f"Error in getmasses: {e}")
            return []

    def getvalence(self):
        try:
            valence = [] 
            for atom in self.matrixdescriptors.element:
                try:
                    valence.append(self.properties.el_valence[atom.GetSymbol()])
                except:
                    valence.append(self.properties.el_valence[atom.GetSymbol().lower()])
            return valence
        except Exception as e:
            print(f"Error in getvalence: {e}")
            return []
    
    def gethyb(self):
        try:
            hyb = []
            for atom in self.matrixdescriptors.element:
                hyb.append(next(key for key, value in atom.GetHybridization().values.items() if value == atom.GetHybridization()))
            return hyb
        except Exception as e:
            print(f"Error in gethyb: {e}")
            return []

    def WtAdjMatByBO(self):
        self.G = (self.matrixdescriptors.bond_mat / 8) * self.matrixdescriptors.adj_mat
    
    def WtAdjMatByAM(self):
        masses = self.getmasses()
        self.G = self.matrixdescriptors.adj_mat * np.sqrt(np.outer(masses, masses))
    
    def WtAdjMatByValence(self):
        valence = self.getvalence()
        self.G = self.matrixdescriptors.adj_mat * np.sqrt(np.outer(valence, valence))
    
    def WtAdjMatByHYB(self):
        hyb = self.gethyb()
        self.G = self.matrixdescriptors.adj_mat * np.sqrt(np.outer(hyb, hyb))

    def WtAdjMatByCoulomb(self):
        try:
            mol = self.matrixdescriptors.new_mol
            coulomb_matrix = self.getCoulombMatrix(mol)
            self.G = self.matrixdescriptors.adj_mat * coulomb_matrix
        except Exception as e:
            print(f"Error in WtAdjMatByCoulomb: {e}")
            self.G = self.matrixdescriptors.adj_mat

    def WtAdjMatByAll(self):
        masses = self.getmasses()
        valence = self.getvalence()
        hyb = self.gethyb()
        try:
            mol = self.matrixdescriptors.new_mol
            coulomb_matrix = self.getCoulombMatrix(mol)
            self.G = self.matrixdescriptors.adj_mat * np.sqrt(np.outer(masses, masses)) + \
                     self.matrixdescriptors.adj_mat * np.sqrt(np.outer(valence, valence)) + \
                     self.matrixdescriptors.adj_mat * np.sqrt(np.outer(hyb, hyb)) + \
                     self.matrixdescriptors.adj_mat * coulomb_matrix
        except Exception as e:
            print(f"Error in WtAdjMatByAll: {e}")
            self.G = self.matrixdescriptors.adj_mat

    def getWeightedMatrix(self):
        if self.params.wt_adj_mat_by == 'bond_order':
            self.WtAdjMatByBO()
        elif self.params.wt_adj_mat_by == 'atomic_mass':
            self.WtAdjMatByAM()
        elif self.params.wt_adj_mat_by == 'valence':
            self.WtAdjMatByValence()
        elif self.params.wt_adj_mat_by == 'hybridization':
            self.WtAdjMatByHYB()
        elif self.params.wt_adj_mat_by == 'coulomb':
            self.WtAdjMatByCoulomb()
        elif self.params.wt_adj_mat_by == 'all':
            self.WtAdjMatByAll()
        else:
            self.G = self.matrixdescriptors.adj_mat
