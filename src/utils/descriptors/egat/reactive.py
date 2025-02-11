# Determines the Reactive Atoms 
# Inputs:       Elements: N-element List strings for each atom type
#               R_bond_mat: Reactant bond order matrix.
#               P_bond_mat: Product bond order matrix.
# Returns:      bond_change: list of bond changes and where they are. 
#               bond_broken: list of bonds broken
#               bond_formed: list of bonds formed
#               bond_ochangeup: list of bonds with the order increased
#               bond_ochangedown: list of bonds with the order increase.
#               involve: Sorted set of bond changes


class BondDescriptor:
    change = []
    formed = []
    broken = []
    ochange = []
    ochangeup = []
    ochangedown = []

class AtomDescriptor:
    reactive = []

class Reactive: 
    def __init__(self,E,Rbond_mat,Pbond_mat):
        self.bondmat_change = Pbond_mat - Rbond_mat
        self.E = E
        self.Rbond_mat = Rbond_mat
        self.Pbond_mat = Pbond_mat
        self.bond = BondDescriptor
        self.atom = AtomDescriptor
        self.GetProperties()
        

    def GetProperties(self):
        for i in range(len(self.E)):
            for j in range(i+1,len(self.E)):
                if self.bondmat_change[i][j] != 0:
                    self.bond.change += [(i,j)]
                    self.atom.reactive += [i,j]
                    # If there was no bond at the reactant, state that the bond is formed. 
                    if self.Rbond_mat[i][j] == 0:
                        self.bond.formed += [(i,j)]
                    # If there was no bond at the product, state that it is broken. 
                    elif self.Rbond_mat[i][j] == 0:
                        self.bond.broken += [(i,j)]
                    elif self.Rbond_mat[i][j] > self.Pbond_mat[i][j]:
                        self.bond.ochangedown += [(i,j)]
                        self.bond.ochange += [(i,j)]
                    elif self.Rbond_mat[i][j] < self.Pbond_mat[i][j]:
                        self.bond.ochangeup += [(i,j)]
                        self.bond.ochange += [(i,j)]

    
    def ReactiveAtoms(self):
        self.atom.reactive = sorted(list(set(self.atom.reactive)))

    def bnfn(self):
        self.bnfn = [len(self.bond.broken),len(self.bond.formed)]

    

