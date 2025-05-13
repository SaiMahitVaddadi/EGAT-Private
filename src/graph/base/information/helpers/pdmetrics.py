from ...base import BaseFeaturizer
from rdkit import Chem
import networkx as nx
import numpy as np


class PathDistanceMetrics(BaseFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)

    def SPDFunc(self,edge):
        try:
            path_length = Chem.GetDistanceMatrix(self.matrixdescriptors.new_mol)[edge[0], edge[1]]
            return [path_length]
        except:
            return [-1]  # Return a large value if the edge does not exist or an error occurs
    
    def getAllPaths(self, start_atom, end_atom, max_length=1000):
        try:
            graph = nx.Graph()
            for bond in self.matrixdescriptors.new_mol.GetBonds():
                graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            
            all_paths = dict()
            all_paths['path'] = []
            all_paths['length'] = []    
            for path in nx.all_simple_paths(graph, source=start_atom, target=end_atom, cutoff=max_length):
                atoms = [self.matrixdescriptors.new_mol.GetAtomWithIdx(int(idx)) for idx in path]
                bonds = []
                for i in range(len(path) - 1):
                    bond = self.matrixdescriptors.new_mol.GetBondBetweenAtoms(path[i], path[i + 1])
                    if bond:
                        bonds.append(bond)
                path_length = len(path) - 1
                all_paths['length'].append(path_length)
                all_paths['path'].append({"atoms": atoms, "bonds": bonds})
            
            if all_paths['length']:
                min_length = min(all_paths['length'])
                min_paths = [all_paths['path'][i] for i, length in enumerate(all_paths['length']) if length == min_length]
                all_paths['shortest_paths'] = min_paths
                all_paths['shortest_length'] = min_length
            return all_paths
        except Exception as e:
            print(f"Error in getAllPaths: {e}")
            return []
    

    def WtSPDByMass(self,edge):
        all_paths = self.getAllPaths(edge[0], edge[1])
        try:
            distance = Chem.GetDistanceMatrix(self.matrixdescriptors.new_mol)[edge[0], edge[1]]
            wdm = []
            for path in all_paths['shortest_paths']:
                atoms = path['atoms']
                total_weight = 0
                for atom in atoms:
                    total_weight += atom.GetMass() / 128
            
                weighted_distance = distance * total_weight
                wdm.append(weighted_distance)
            return [min(wdm)]
        except Exception as e:
            print(f"Error in WtSPDByMass: {e}")
            return [-1]

    def WtSPDByBO(self,edge):
        all_paths = self.getAllPaths(edge[0], edge[1])
        try:
            distance = Chem.GetDistanceMatrix(self.matrixdescriptors.new_mol)[edge[0], edge[1]]
            wdm = []
            for path in all_paths['shortest_paths']:
                bonds = path['bonds']
                total_weight = 0
                for bond in bonds:
                    total_weight += bond.GetBondTypeasDouble() / 8
            
                weighted_distance = distance * total_weight
                wdm.append(weighted_distance)
            weighted_distance = min(wdm)
            return [weighted_distance]
        except Exception as e:
            print(f"Error in WtSPDByMass: {e}")
            return [-1]


    def WTSPDByHYB(self,edge):
        all_paths = self.getAllPaths(edge[0], edge[1])
        try:
            distance = Chem.GetDistanceMatrix(self.matrixdescriptors.new_mol)[edge[0], edge[1]]
            wdm = []
            for path in all_paths['shortest_paths']:
                atoms = path['atoms']
                total_weight_atoms = 0
                total_weight_hyb = 0
                for atom in atoms:
                    total_weight_atoms += atom.GetMass() / 128
                    hyb = next(key for key, value in atom.GetHybridization().values.items() if value == atom.GetHybridization())
                    total_weight_hyb += hyb / 9
                
                total_weight = total_weight_atoms + total_weight_hyb
                wdm.append(total_weight*distance)
            return [min(wdm)]
        except Exception as e:
            print(f"Error in WtSPDByMass: {e}")
            return [-1]
        

    def WTSPDByVE(self,edge):
        all_paths = self.getAllPaths(edge[0], edge[1])
        try:
            distance = Chem.GetDistanceMatrix(self.matrixdescriptors.new_mol)[edge[0], edge[1]]
            wdm = []
            for path in all_paths['shortest_paths']:
                atoms = path['atoms']
                total_weight_atoms = 0
                total_weight_hyb = 0
                for atom in atoms:
                    total_weight_atoms += atom.GetMass() / 128
                    try:
                        total_weight_hyb += self.properties.el_valence[atom.GetSymbol()] / 8
                    except: 
                        total_weight_hyb += self.properties.el_valence[atom.GetSymbol().lower()] / 8
                total_weight = total_weight_atoms + total_weight_hyb
                wdm.append(total_weight*distance)
            return [min(wdm)]
        except Exception as e:
            print(f"Error in WtSPDByMass: {e}")
            return [-1]
        
    def WTSPDByHYBVE(self,edge):
        all_paths = self.getAllPaths(edge[0], edge[1])
        try:
            distance = Chem.GetDistanceMatrix(self.matrixdescriptors.new_mol)[edge[0], edge[1]]
            wdm = []
            for path in all_paths['shortest_paths']:
                atoms = path['atoms']
                total_weight_atoms = 0
                total_weight_hyb = 0
                total_weight_ve = 0
                for atom in atoms:
                    total_weight_atoms += atom.GetMass() / 128
                    hyb = next(key for key, value in atom.GetHybridization().values.items() if value == atom.GetHybridization())
                    total_weight_hyb += hyb / 9
                    try:
                        total_weight_ve += self.properties.el_valence[atom.GetSymbol()] / 8
                    except: 
                        total_weight_ve += self.properties.el_valence[atom.GetSymbol().lower()] / 8
                
                total_weight = total_weight_atoms + total_weight_hyb + total_weight_ve
                wdm.append(total_weight*distance)
            return [min(wdm)]
        except Exception as e:
            print(f"Error in WtSPDByMass: {e}")
            return [-1]

    def getCoulombMatrix(self, mol,id=1):
        try:
            num_atoms = mol.GetNumAtoms()
            coulomb_matrix = np.zeros((num_atoms, num_atoms))

            for i in range(num_atoms):
                atom_i = mol.GetAtomWithIdx(int(i))
                Z_i = atom_i.GetAtomicNum()
                pos_i = np.array(mol.GetConformer(conf_id=id).GetAtomPosition(i))

                for j in range(num_atoms):
                    if i == j:
                        coulomb_matrix[i, j] = 0.5 * Z_i ** 2.4  # Diagonal elements
                    else:
                        atom_j = mol.GetAtomWithIdx(int(j))
                        Z_j = atom_j.GetAtomicNum()
                        pos_j = np.array(mol.GetConformer(conf_id=id).GetAtomPosition(j))
                        distance = np.linalg.norm(pos_i - pos_j)
                        coulomb_matrix[i, j] = (Z_i * Z_j) / distance  # Off-diagonal elements

            return coulomb_matrix
        except Exception as e:
            print(f"Error in getCoulombMatrix: {e}")
            return np.zeros((0, 0))
        

    def WTSPDbyCM(self,edge):
        try:
            mol = self.matrixdescriptors.new_mol
            coulomb_matrix = self.getCoulombMatrix(mol)
            distance = Chem.GetDistanceMatrix(mol)[edge[0], edge[1]]
            weight = coulomb_matrix[edge[0], edge[1]]
            return [distance * weight]
        except Exception as e:
            print(f"Error in WTSPDbyCM: {e}")
            return [-1]


    def usePBC(self,edge,id=1):
        mol = self.matrixdescriptors.new_mol
        try:
            pos1 = mol.GetConformer(conf_id=id).GetAtomPosition(edge[0])
            pos2 = mol.GetConformer(conf_id=id).GetAtomPosition(edge[1])
            delta = pos2 - pos1
            delta -= self.params.sp_box_size * np.round(delta / self.params.sp_box_size)  # Apply periodic boundary conditions
            distance = np.linalg.norm(delta)
            return [distance]
        except:
            return [-1]
            

    def WtSPDByAll(self,edge):
        all_paths = self.getAllPaths(edge[0], edge[1])
        try:
            distance = Chem.GetDistanceMatrix(self.matrixdescriptors.new_mol)[edge[0], edge[1]]
            wdm = []
            for path in all_paths['shortest_paths']:
                atoms = path['atoms']
                bonds = path['bonds']
                total_weight_bonds = 0
                total_weight_atoms = 0
                total_weight_hyb = 0
                total_weight_ve = 0
                for atom in atoms:
                    total_weight_atoms += atom.GetMass() / 128
                    hyb = next(key for key, value in atom.GetHybridization().values.items() if value == atom.GetHybridization())
                    total_weight_hyb += hyb / 9
                    try:
                        total_weight_ve += self.properties.el_valence[atom.GetSymbol()] / 8
                    except: 
                        total_weight_ve += self.properties.el_valence[atom.GetSymbol().lower()] / 8
                for bond in bonds:
                    total_weight_bonds += bond.GetBondTypeasDouble() / 8

                total_weight = total_weight_atoms + total_weight_bonds + total_weight_hyb + total_weight_ve
                wdm.append(total_weight*distance)
            return [min(wdm)]
        except Exception as e:
            print(f"Error in WtSPDByMass: {e}")
            return [-1]

