from ast import excepthandler
import sys
import h5py
sys.path.append('../utilities')
from taffi_functions import adjmat_to_adjlist,graph_seps,xyz_parse,find_lewis,return_ring_atom
from utility import *
from yarpecule import return_rings
import json
import os,sys,subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cdist
import argparse 
import joblib
from joblib import Parallel,delayed
import traceback    
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import rdDistGeom
from rdkit.Chem import inchi
from rxnmapper import RXNMapper
import yarp as yp
# Determines the locations of Lone Pair Atoms from smiles 
# Inputs:       SMILES: Smiles string of molecule.
# Returns:      lone_pairs: a set of indices with radicals in them. 
def return_lonepairs_RDKit(smiles):
    # load in mol
    mol = Chem.MolFromSmiles(smiles,sanitize=False)

    rdDistGeom.EmbedMolecule(mol,useRandomCoords=True,randomSeed=42)
    lp_atom_index = []
    amap = []

    for atom in mol.GetAtoms():
        amap += [atom.GetAtomMapNum()]
        radicals = atom.GetNumRadicalElectrons()
        Hs = atom.GetTotalNumHs()
        bonds = 0
        for neighbor in atom.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            if bond.GetIsAromatic():
                bonds += 2.0
            else:
                bonds += bond.GetBondTypeAsDouble()

        if atom.GetSymbol() == 'H':
            valence = 1
        elif atom.GetSymbol() == 'P':
            valence = 10
        elif atom.GetSymbol() == 'S':
            valence = 12
        else:
            valence = 8
        
        lp = valence - bonds - Hs - radicals
        
        lp_atom_index += [lp]
    
    result = pd.DataFrame({'lp_atom_index':lp_atom_index,'mapping':amap})
    result = result.sort_values('mapping')
    lps = result['lp_atom_index'].tolist()
    return lps

# Determines the locations of Lone Pair Atoms from smiles 
# Inputs:       SMILES: Smiles string of molecule.
# Returns:      lone_pairs: a set of indices with radicals in them. 
def return_radicals_RDKit(smiles):
    # load in mol
    mol = Chem.MolFromSmiles(smiles,sanitize=False)

    rdDistGeom.EmbedMolecule(mol,useRandomCoords=True,randomSeed=42)
    
    radical_atom_index = []
    amap = []

    for atom in mol.GetAtoms():
        radical_atom_index += [atom.GetNumRadicalElectrons()]
        amap += [atom.GetAtomMapNum()]
    result = pd.DataFrame({'radical_atom_index':radical_atom_index,'mapping':amap})
    result = result.sort_values('mapping')
    radicals = result['radical_atom_index'].tolist()
    return radicals


def getradicalsYARP(smiles):
    cule = yp.yarpecule(smiles,canon=False,mapping=True)
    bond_mat = cule.bond_mats[0]
    fc = cule.fc

    lpmatrix = []
    radmatrix = []
    diradmatrix = []

    for i in range(bond_mat.shape[0]): 
        lpmatrix += [bond_mat[i,i]]
        row = bond_mat[i,:]
        res = 0 
        for _,bond in enumerate(row):
            if _ != i:
                res += 2*bond
            else:
                res += bond
        if res < 8:
            radmatrix += [bond_mat[i,i]]
        else:
            if bond_mat[i,i] % 2 == 1:
                if fc < 0:
                    radmatrix += [0]
                else:
                    radmatrix += [1]
            else:
                radmatrix += [0]
    return lpmatrix,radmatrix



