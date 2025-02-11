from .molmatdesc import MolMatDesc
from rdkit import Chem
from rdkit.Chem import rdDistGeom
import pandas as pd 
import yarp as yp
from .properties import Properties
class RDKElectronInfo:
    def __init__(self,egatecule:MolMatDesc):
        self.new_mol = egatecule.new_mol
        self.props = Properties()
        rdDistGeom.EmbedMolecule(self.new_mol,useRandomCoords=True,randomSeed=42)
        self.lp_atom_index = dict()
        self.amap = dict()

    def GetValence(self,atom):
        symbol = atom.GetSymbol()
        self.valence = self.props.el_valence[symbol]
        self.alt_valence = self.props.el_alt_valence[symbol]
    
    def GetBonds(self,atom):
        bonds = 0
        for neighbor in atom.GetNeighbors():
            bond = self.new_mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            if bond.GetIsAromatic():
                bonds += 2.0
            else:
                bonds += bond.GetBondTypeAsDouble()
        return bonds
    
    def EncodeLonePair(self,atom):
        amap = atom.GetAtomMapNum()
        radicals = atom.GetNumRadicalElectrons()
        formal_charge = atom.GetFormalCharge()
        self.GetValence(atom)
        lp = self.valence - self.GetBonds(atom) - atom.GetNumRadicalElectrons() - atom.GetFormalCharge()
        lp = self.ChangeForSpecialCases(lp,atom)
        return amap,lp,radicals

    def ChangeForSpecialCases(self,lp,atom):
        if atom.GetSymbol() in ['P','S'] and lp < 0:
            lp = self.alt_valence - self.GetBonds(atom) - atom.GetNumRadicalElectrons() - atom.GetFormalCharge()
            if lp < 0: lp = 0
        else:
            if lp < 0 : lp = 0
        return lp

    def GetElectronInfo(self):
        amaplist = []
        lplist = []
        radlist = [] 
        for atom in self.new_mol.GetAtoms():
            amap,lp,radicals = self.EncodeLonePair(atom)
            amaplist.append(amap)
            lplist.append(lp)
            radlist.append(radicals)

        result = pd.DataFrame({'lp_atom_index':lplist,'mapping':amaplist,'radical_atom_index':radlist})
        result = result.sort_values('mapping')
        self.lps = result['lp_atom_index'].tolist()
        self.rads = result['lp_atom_index'].tolist()
        
    def __call__(self):
        self.GetElectronInfo()


class YARPElectronInfo: 
    def __init__(self,egatecule:MolMatDesc,canon:bool = True,mapping:bool = True):
        self.new_mol = egatecule.new_mol
        self.yarpecule = yp.yarpecule(egatecule.new_smiles,canon,mapping) # this is a neat workaround, but we should see if we can just dump the molecule instead. 
        self.lps = []
        self.rads = []

    # check if we can pull atoms from the yarpecule
    def GetValence(self,atom):
        symbol = atom.GetSymbol()
        self.valence = self.props.el_valence[symbol]
        self.alt_valence = self.props.el_alt_valence[symbol]
    
    def ValenceCheck(self,nbelectrons,bonds,fc,valence):
        lp = valence - fc - bonds - nbelectrons
        return lp

    def EncodeRadicalInfo(self,bm,i):
        row = bm[i,:]
        atom = self.new_mol.GetAtoms()[i]
        self.GetValence(atom)
        lp = self.ValenceCheck(bm[i,i],self.GetBonds(row,i),self.yarpecule.fc[i],self.valence)
        lp = self.ChangeForSpecialCases(lp,atom,bm,row,i)
        return lp

    def ChangeForSpecialCases(self,lp,atom,bm,row,i):
        if atom.GetSymbol() in ['P','S'] and lp < 0:
            lp = self.ValenceCheck(bm[i,i],self.GetBonds(row,i),self.yarpecule.fc[i],self.alt_valence)
            if lp < 0: lp = bm[i,i]
        else:
            if lp < 0 : lp = bm[i,i]
        return lp
                
    def GetBonds(self,row,i):
        bonds = 0 
        for _,bond in enumerate(row):
            if _ != i:
                bonds += 2*bond
        return bonds
        
    def GetElectronInfo(self):
        for ind,bm in enumerate(self.yarpecule.bond_mats):
            self.lps[ind] = []
            self.rads[ind]= []
            for i in range(bm.shape[0]):
                self.lps[ind].append(bm[i,i])
                self.rads[ind].append(self.EncodeRadicalInfo(bm,i))
    
    def __call__(self):
        self.GetElectronInfo()



