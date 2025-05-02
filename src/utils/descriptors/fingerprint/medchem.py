import random
from typing import List, Tuple, Optional

import datamol as dm
import medchem as mc
from medchem.catalogs import NamedCatalogs
from medchem.catalogs import list_named_catalogs
from medchem.catalogs import catalog_from_smarts


class MedChem:
    def __init__(self,smiles):
        self.smiles = smiles
        if isinstance(smiles, list):
            self.mol = [dm.to_mol(s) for s in smiles]
            self.molblock = [dm.to_molblock(m) for m in self.mol]
            self.mol2d = [dm.to_mol2d(m) for m in self.mol]
            self.mol3d = [dm.to_mol3d(m) for m in self.mol]
            self.mol3d_opt = [dm.to_mol3d(m, optimize=True) for m in self.mol]
            self.mol3d_opt_block = [dm.to_molblock(m) for m in self.mol3d_opt]
            self.mol3d_opt_2d = [dm.to_mol2d(m) for m in self.mol3d_opt]
            self.mol3d_opt_smiles = [dm.to_smiles(m) for m in self.mol3d_opt]
            self.mol3d_opt_inchi = [dm.to_inchi(m) for m in self.mol3d_opt]
            self.mol3d_opt_inchikey = [dm.to_inchikey(m) for m in self.mol3d_opt]
        else:
            self.mol = dm.to_mol(smiles)
            self.molblock = dm.to_molblock(self.mol)
            self.mol2d = dm.to_mol2d(self.mol)
            self.mol3d = dm.to_mol3d(self.mol)
            self.mol3d_opt = dm.to_mol3d(self.mol, optimize=True)
            self.mol3d_opt_block = dm.to_molblock(self.mol3d_opt)
            self.mol3d_opt_2d = dm.to_mol2d(self.mol3d_opt)
            self.mol3d_opt_smiles = dm.to_smiles(self.mol3d_opt)
            self.mol3d_opt_inchi = dm.to_inchi(self.mol3d_opt)
            self.mol3d_opt_inchikey = dm.to_inchikey(self.mol3d_opt)
        
    def max_ring_filter(self,max_ring_size=12):
        return mc.functional.macrocycle_filter(self.mol,max_ring_size=max_ring_size)
    
    def nibr_filter(self,severity=12):
        return mc.functional.nibr_filter(self.mol,max_severity=severity)

    def atom_filter(self,atoms=['C'],max_atoms=10):
        return mc.functional.atom_filter(self.mol,atoms,max_atoms=max_atoms)
    
    def min_ring_filter(self,min_ring_size=3):
        return mc.functional.ring_infraction_filter(self.mol,min_ring_size=min_ring_size)
    
    def atomsize_filter(self,atomrange=[6,10]):
        return mc.functional.num_atom_filter(self.mol, min_atoms=atomrange[0], max_atoms=atomrange[1])
    
    def stereocenter_filter(self,max_stereo_centers=4, max_undefined_stereo_centers=2):
        return mc.functional.stereocenter_filter(self.mol,max_stereo_centers=max_stereo_centers,max_undefined_stereo_centers=max_undefined_stereo_centers)
    
    def halogen_filter(self,thresh_F=6, thresh_Br=3, thresh_Cl=3):
        return mc.functional.halogen_filter(self.mol,thresh_F=thresh_F,thresh_Br=thresh_Br,thresh_Cl=thresh_Cl)
    
    def symmetry_filter(self,symmetry_threshold=0.8):
        return mc.functional.symmetry_filter(self.mol,symmetry_threshold=symmetry_threshold)
    
    def catalog_filter(self,catalogs=['tox']):
        return mc.functional.catalog_filter(self.mol,catalogs)
    
    def chemgroup_filter(self,group='common_oragnic_solvents'):
        cg = mc.groups.ChemicalGroup(group)
        return mc.functional.functional.chemical_group_filter(self.mol,cg)
    
    def rule_filter(self,rule = 'rule_of_five'):
        rfilter = mc.rules.RuleFilters(rule_list=[rule])
        return mc.functional.rules_filter(self.mol,rfilter)
    
    def custom_catalog_filter(self,names,smarts):
        custom_catalog = catalog_from_smarts(smarts=smarts,labels=names,entry_as_inds=False)
        return [custom_catalog.HasMatch(x) for x in self.mol]

    def bredt_filter(self):
        return mc.functional.bredt_filter(self.mol)
    
    def molecular_graph_filter(self,severity=12):
        mc.functional.molecular_graph_filter(self.mol,severity)
    
    def lilly_filter(self):
        return mc.functional.lilly_demerit_filter(self.mol)

    def protecting_groups_filter(self,groups=['fmoc']):
        return mc.functional.protecting_groups_filter(self.mol,groups)

    def complexity_filter(self,complexity_metric='bertz',threshold_stats_file='zinc_15_available'):
        return mc.functional.complexity_filter(self.mol,complexity_metric,threshold_stats_file)

    def alert_filter(self,alerts=['BMS'],alert_db=None):
        if alert_db is None:
            mc.functional.alert_filter(self.mol,alerts)
        else:
            mc.functional.alert_filter(self.mol,alerts,alert_db)
    





