from ..utils.descriptors.egat.encodings import Encodings
from ..utils.descriptors.egat.reactive import Reactive
from ..utils.descriptors.egat.molmatdesc import MolMatDesc
from ..utils.descriptors.egat.radicals import RDKElectronInfo,YARPElectronInfo
from ..utils.descriptors.egat.stereo import StereoChemistry
from ..utils.matrices.graph_seps import graph_seps
from ..utils.misc.taffi_functions import return_rings,adjmat_to_adjlist
from ..utils.descriptors.geometry.geometry import ConformerGenerator

from rdkit import Chem

class ReactionComponentFeaturizerwithPaddingandGeometry:
    def __init__(self, smiles, arguments):
        self.smiles = smiles
        self.params = arguments
        self.properties = Encodings()
        self.CreateAtomMapping()
        self.MatrixDescriptors()
        self.Rings()
        self.Eneg()

    def Eneg(self):
        with open('../utils/database/pauling.txt', 'r') as file:
            pauling_data = file.readlines()
        
        self.pauling_dict = {}
        for line in pauling_data:
            element, electronegativity = line.strip().split()
            self.pauling_dict[element] = float(electronegativity)

    
    def CreateAtomMapping(self):
        molecule = Chem.MolFromSmiles(self.smiles)
        molecule = Chem.AddHs(molecule)
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)
        self.am_smiles = Chem.MolToSmiles(molecule)
    
    def MatrixDescriptors(self):
        self.matrixdescriptors = MolMatDesc(self.am_smiles)
        self.matrixdescriptors.run()
        
    def DistanceMatrix(self):
        self.gs = graph_seps(self.matrixdescriptors.adj_mat)
        self.gs[self.gs < 0] = 100

    def Rings(self):
        self.ring_atoms = return_rings(adjmat_to_adjlist(self.adj),max_size=20,remove_fused=True)
    
    def Radicals(self):
        if self.params.getradical == 'RDKit':
            self.electroninfo = RDKElectronInfo(self.matrixdescriptors)
        elif self.params.getradical == 'YARP':
            self.electroninfo = YARPElectronInfo(self.matrixdescriptors)

    def Rotatability(self):
        self.matrixdescriptors.BridgeHead()
        self.matrixdescriptors.Spiro()
        self.matrixdescriptors.RotatableBondCount()
    
    def Polarity(self):
        self.matrixdescriptors.BondPolarityPauling()
        self.matrixdescriptors.Electronegativity()
    
    def Charges(self):
        self.matrixdescriptors.Gasteiger()
        self.matrixdescriptors.BondDipoleMoments()

    def Stereochem(self):
        self.stereo = StereoChemistry(self.matrixdescriptors)
        self.stereo.run()


    def CheckMapping(self,ind):
        atom_mappings = min([atom.GetAtomMapNum() for atom in self.new_mol.GetAtoms()])
        #print(f"ind: {ind}, Rsmiles: {Rsmiles}, molecule: {molecule}, Generated Atom Mapping: {atom_mappings}\n")
        if atom_mappings == 0:
            ind_in_mol = ind
        else:
            ind_in_mol = ind+1
        return ind

    def EncodeElement(self,ind):
        if not self.params.removeelementinfo:
            return self.properties.element_encode[self.matrixdescriptors.element[ind]]
        else:
            return []

    def GetNeighbors(self,ind):
        neighbors = [self.matrixdescriptors.element[counti] for counti,i in enumerate(self.matrixdescriptors.adj_mat[ind,:]) if i != 0]
        if self.params.neighbor == 'onlyH':
            NE_count = [neighbors.count('H')]
        elif self.params.neighbor == 'onlyCHNO':
            NE_count = [neighbors.count('H'), neighbors.count('C'), neighbors.count('N'), neighbors.count('O')]
        elif self.params.neighbor == 'onlyorganic':
            NE_count = [neighbors.count('H'), neighbors.count('C'), neighbors.count('N'), neighbors.count('O'), 
                        neighbors.count('P'), neighbors.count('S'), neighbors.count('F'), neighbors.count('Cl'), 
                        neighbors.count('Br'), neighbors.count('I')]
        elif self.params.neighbor == 'all':
            NE_count = [neighbors.count(element) for element in self.properties.element_encode.keys()]
        else:
            NE_count = []
        return NE_count

    def DistanceFromReactingAtom(self,ind):
        if not self.params.removereactiveinfo:
            if len(self.reactive_atoms) > 0:
                dis = min([self.gs[ind][indr] for indr in self.reactive_atoms])
            else:
                dis = 0 
            return [dis]
        else:
            return []

    def RingCheck(self,ind):
        if not self.params.removeringinfo:
            if True in [ind in ra_list for ra_list in self.ring_atoms]: inR = 1
            else: inR = 0
            return [inR]
        else:
            return []
    
    def FormalCharge(self,ind):
        if not self.params.removeformalchargeinfo:
            fc = self.matrixdescriptor.fc[ind]
            return fc
        else:
            return []

    def AromaticityCheck(self,ind):
        if not self.args.removearomaticity:
            ###### GET AROMATICITY
            if self.self.matrixdescriptors.element[ind] == 'H': aromaticity = 0
            elif self.stereo.atom_aromatic[ind]: aromaticity = 1
            else: aromaticity = 0
            return [aromaticity]
        else:
            return []
       
    def HybridizationCheck(self,ind):
        ###### GET HYBRIDIZATION
        if not self.args.removehybridinfo:
            if self.self.matrixdescriptors.element[ind] == 'H':
                if not self.params.useFullHyb: hybrid = [0,0,0,1]
                else: hybrid = [0,0,0,1,0,0,0,0,0]

            else:
                hybrid = self.stereo.Hybridization[ind] 
            return hybrid
        else:
            return []
    
    def ChiralityCheck(self,ind):
        if not self.args.removechiralinfo:
            ###### CHECK IF IT IS IN A CHIRAL CENTER
            if ind in self.stereo.chiral_centers: chiral = self.properties.atom_chiral_encode[self.stereo.chiral_centers[ind]]
            else: chiral = [0,0,1]
            return [chiral]
        else:
            return []
    
    def RadicalCheck(self,ind):
        if self.args.getradical:
            return [self.electroninfo.rads[ind],self.electroninfo.lps[ind]]
        else:
            return []

    def SpiroCheck(self,ind):
        if self.args.getspiro:
            return [self.matrixdescriptors.spiro[ind]]
        else:
            return []

    def BridgeHeadCheck(self,ind):
        if self.args.getbridgehead:
            return [self.matrixdescriptors.bridgehead[ind]]
        else:
            return []

    def ElectronegativityCheck(self,ind):
        if self.args.getelectronegativity:
            return [self.matrixdescriptors.electronegativity[ind]]
        else:
            return []

    def ChargeCheck(self,ind):
        if self.args.charge == 'Gasteiger':
            return [self.matrixdescriptors.gasteiger_charges[ind]]
        elif self.args.charge == 'psi4':
            return []
        elif self.args.charge == 'pyscf':
            return []
        elif self.args.charge == 'gaussian':
            return []
        elif self.args.charge == 'orca':
            return []
        elif self.args.charge == 'ase':
            return []
        elif self.args.charge == 'gemnet':
            return []
        elif self.args.charge == 'chemprop':
            return []
        elif self.args.charge == 'unimol':
            return []
        else:
            return []

    def GenerateAtomFeatureVector(self):
        self.atom_features = []
        self.node_vector_length = 0 
        for ind in range(len(self.matrixdescriptors.element)):
            atom_feature = []
            atom_feature += self.EncodeElement(ind)
            atom_feature += self.GetNeighbors(ind)
            atom_feature += self.RingCheck(ind)
            atom_feature += self.FormalCharge(ind)
            atom_feature += self.AromaticityCheck(ind)
            atom_feature += self.HybridizationCheck(ind)
            atom_feature += self.ChiralityCheck(ind)
            atom_feature += self.RadicalCheck(ind)
            atom_feature += self.SpiroCheck(ind)
            atom_feature += self.BridgeHeadCheck(ind)
            atom_feature += self.ElectronegativityCheck(ind)
            self.atom_features.append(atom_feature)
            self.node_vector_length = len(atom_feature)


    def InitializeAddons(self):
        self.Radicals()
        self.Rotatability()
        self.Charges()
        self.Polarity()
        self.Stereochem()
    
    def GenerateEdges(self):
        ###### GENERATE THE RP-ADJACENCY MATRIX SO THAT AT LEAST ONE SIDE IS CONNECTED 
        self.edges_u,self.edges_v  = [],[]
        for i in range(len(self.matrixdescriptors.element)):
            for j in range(len(self.matrixdescriptors.element)):
                # if reaction, also check if P_adj > 0
                if self.matrixdescriptors.adj_mat[i][j] > 0:
                    self.edges_u.append(i)
                    self.edges_v.append(j)
        
    
    def EncodeBondOrder(self,edge):
        if self.EdgeCheck(edge):
            BO = self.matrixdescriptors.bond_mat[edge[0],edge[1]]
            return BO
        else:
            return 0

    def EdgeCheck(self,edge):
        if self.matrixdescriptors.adj_mat[edge[0]][edge[1]] > 0:
            return True
        elif self.matrixdescriptors.adj_mat[edge[1]][edge[0]] == 0:
            return False
        else:
            return None

    def BondinRing(self,edge):
        if self.EdgeCheck(edge):
            if not self.params.removeringinfo:
                if True in [(edge[0] in ra_list and edge[1] in ra_list) for ra_list in self.ring_atoms]: inR = 1
                else: inR = 0
                return [inR]
            else:
                return []
        else:
            if not self.params.removeringinfo:
                return [0]
            else:
                return []
            
    
    def BondOrder(self,edge,bo):
        if self.EdgeCheck(edge):
            if not self.params.removebondorderinfo:
                if bo == 0:            
                    return [0,0,0,0,1]
                else:
                    if tuple(edge) in self.stereo.bond_aromatic and self.stereo.bond_aromatic[tuple(edge)]: return self.properties.bond_order_encode['BA']
                    else: return self.properties.bond_order_encode['B{}'.format(int(bo))]
            else:
                return []
        else:
            if not self.params.removebondorderinfo:
                return [0,0,0,0,1]
            else:
                return []

    def BondConjugation(self,edge,bo):
        if self.EdgeCheck(edge):
            # Add Bond Conjugation info if not stated that you want to remove it from training. 
            if not self.params.removeconjinfo:
                if bo == 0:
                    return [0]
                else:
                    if tuple(edge) in self.stereo.conjugation and self.stereo.conjugation[tuple(edge)]: return [1]
                    else: return [0]
            else:
                return []
        else:
            if not self.params.removeconjinfo:
                return [0]
            else:
                return []
        
    def BondStereochemistry(self,edge,bo):
        if self.EdgeCheck(edge):
            # Add Bond Conjugation info if not stated that you want to remove it from training. 
            if not self.params.removestereoinfo:
                if bo == 0:
                    return [0,0,0]
                else:
                    if tuple(edge) in self.stereo.bond_stereo and self.stereo.bond_stereo[tuple(edge)]: return self.properties.bond_stereo_encode[self.stereo.bond_stereo[tuple(edge)]]
                    else: return [0,0,0]
            else:
                return []
        else:
            if not self.params.removestereoinfo:
                return [0,0,0]
            else:
                return []
    
    def BondRotation(self,edge,bo):
        if self.EdgeCheck(edge):
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
        else:
            if self.params.getrotatablebonds:
                return [0,0]
            else:
                return []
        

    def GenerateConformers(self):
        # Fix for each 3D Geometry Tools That's There
        self.conformers = ConformerGenerator(self.matrixdescriptors,self.params.conformer.nconfs,self.params.conformer.method,
                                             self.params.conformer.seed,self.params.conformer.verbose,self.params.conformer.rdkoptimizer)

        confgenerator = getattr(self.conformers,f'Generatewith{self.params.conformer.generator}',None)

        if self.params.conformer.generator == 'RDKit':
            confgenerator(self.params.conformer.seed)
        elif self.params.conformer.generator == 'Auto3D':
            confgenerator(self.params.conformer.seed)
        elif self.params.conformer.generator == 'ASE':
            confgenerator(self.params.conformer.seed)
        elif self.params.conformer.generator == 'CREST':
            confgenerator(self.params.conformer.seed)
        elif self.params.conformer.generator == 'CREGEN':
            confgenerator(self.params.conformer.seed)
        elif self.params.conformer.generator == 'QCG':
            confgenerator(self.params.conformer.seed)
        elif self.params.conformer.generator == 'OpenFF':
            confgenerator(self.params.conformer.seed)
        elif self.params.conformer.generator == 'OpenMM':
            confgenerator(self.params.conformer.seed)
        elif self.params.conformer.generator == 'TorsDiff':
            confgenerator(self.params.conformer.seed)
        
        self.matrixdescriptors.new_mol = self.conformers.egatecule.new_mol

    def FindHBonds(self):
        self.hbond_edges = []
        for donor in range(len(self.matrixdescriptors.element)):
            if self.matrixdescriptors.element[donor] in ['O', 'N', 'F', 'Cl', 'Br', 'I']:
                for hydrogen in range(len(self.matrixdescriptors.element)):
                    if self.matrixdescriptors.element[hydrogen] == 'H' and self.matrixdescriptors.adj_mat[donor][hydrogen] > 0:
                        for acceptor in range(len(self.matrixdescriptors.element)):
                            if self.matrixdescriptors.element[acceptor] in ['O', 'N', 'F', 'Cl', 'Br', 'I'] and donor != acceptor:
                                distance = self.matrixdescriptors.new_mol.GetConformer().GetAtomPosition(hydrogen).Distance(self.matrixdescriptors.new_mol.GetConformer().GetAtomPosition(acceptor))
                                if distance > 0 and distance <= 3:  # typical H-bond distance cutoff
                                    self.hbond_edges.append([hydrogen, acceptor])
    
    def FindElectrostaticInteractions(self):
        self.electrostatic_edges = dict()
        self.electrostatic_edges['attract'] = []
        self.electrostatic_edges['repel'] = []
        for atom1 in range(len(self.matrixdescriptors.element)):
            for atom2 in range(atom1 + 1, len(self.matrixdescriptors.element)):
                if self.matrixdescriptors.fc[atom1] != 0 and self.matrixdescriptors.fc[atom2] != 0:
                    distance = self.matrixdescriptors.new_mol.GetConformer().GetAtomPosition(atom1).Distance(self.matrixdescriptors.new_mol.GetConformer().GetAtomPosition(atom2))
                    if distance > 0 and distance <= 5:  # typical electrostatic interaction distance cutoff
                        if self.matrixdescriptors.fc[atom1] > 0 and self.matrixdescriptors.fc[atom2] < 0 or self.matrixdescriptors.fc[atom1] < 0 and self.matrixdescriptors.fc[atom2] > 0:
                            self.electrostatic_edges['attract'].append([atom1, atom2])
                        elif self.matrixdescriptors.fc[atom1] > 0 and self.matrixdescriptors.fc[atom2] > 0 or self.matrixdescriptors.fc[atom1] < 0 and self.matrixdescriptors.fc[atom2] < 0:
                            self.electrostatic_edges['repel'].append([atom1, atom2])

    def FindCH_OInteractions(self):
        self.ch_o_edges = []
        for carbon in range(len(self.matrixdescriptors.element)):
            if self.matrixdescriptors.element[carbon] == 'C':
                for hydrogen in range(len(self.matrixdescriptors.element)):
                    if self.matrixdescriptors.element[hydrogen] == 'H' and self.matrixdescriptors.adj_mat[carbon][hydrogen] > 0:
                        for oxygen in range(len(self.matrixdescriptors.element)):
                            if self.matrixdescriptors.element[oxygen] == 'O' and carbon != oxygen:
                                distance = self.matrixdescriptors.new_mol.GetConformer().GetAtomPosition(hydrogen).Distance(self.matrixdescriptors.new_mol.GetConformer().GetAtomPosition(oxygen))
                                if distance > 0 and distance <= 3:  # typical C–H···O interaction distance cutoff
                                    self.ch_o_edges.append([hydrogen, oxygen])

    def FindDihydrogenBonds(self):
        self.dihydrogen_edges = []
        for hydrogen1 in range(len(self.matrixdescriptors.element)):
            if self.matrixdescriptors.element[hydrogen1] == 'H':
                for hydrogen2 in range(hydrogen1 + 1, len(self.matrixdescriptors.element)):
                    if self.matrixdescriptors.element[hydrogen2] == 'H':
                        distance = self.matrixdescriptors.new_mol.GetConformer().GetAtomPosition(hydrogen1).Distance(self.matrixdescriptors.new_mol.GetConformer().GetAtomPosition(hydrogen2))
                        if distance > 0 and distance <= 2.5:  # typical dihydrogen bond distance cutoff
                            self.dihydrogen_edges.append([hydrogen1, hydrogen2])
    
    def FindCationPiInteractions(self):
        self.cation_pi_edges = []
        for cation in range(len(self.matrixdescriptors.element)):
            if self.matrixdescriptors.fc[cation] > 0:
                for ring in self.ring_atoms:
                    if all(self.matrixdescriptors.element[atom] in ['C', 'N', 'O', 'S'] for atom in ring):  # check if all atoms in the ring are aromatic
                        for atom in ring:
                            distance = self.matrixdescriptors.new_mol.GetConformer().GetAtomPosition(cation).Distance(self.matrixdescriptors.new_mol.GetConformer().GetAtomPosition(atom))
                            if distance > 0 and distance <= 5:  # typical cation–π interaction distance cutoff
                                self.cation_pi_edges.append([cation, atom])

    def FindHalogenBonds(self):
        self.halogen_bond_edges = []
        halogens = ['F', 'Cl', 'Br', 'I']
        for halogen in range(len(self.matrixdescriptors.element)):
            if self.matrixdescriptors.element[halogen] in halogens:
                for acceptor in range(len(self.matrixdescriptors.element)):
                    if self.matrixdescriptors.element[acceptor] in ['O', 'N', 'S'] and halogen != acceptor:
                        distance = self.matrixdescriptors.new_mol.GetConformer().GetAtomPosition(halogen).Distance(self.matrixdescriptors.new_mol.GetConformer().GetAtomPosition(acceptor))
                        if distance > 0 and distance <= 3.5:  # typical halogen bond distance cutoff
                            self.halogen_bond_edges.append([halogen, acceptor])

    def FindMetallophilicInteractions(self):
        self.metallophilic_edges = []
        metals = ['Cu', 'Ag', 'Au', 'Zn', 'Cd', 'Hg', 'Pt', 'Pd']
        for metal1 in range(len(self.matrixdescriptors.element)):
            if self.matrixdescriptors.element[metal1] in metals:
                for metal2 in range(metal1 + 1, len(self.matrixdescriptors.element)):
                    if self.matrixdescriptors.element[metal2] in metals:
                        distance = self.matrixdescriptors.new_mol.GetConformer().GetAtomPosition(metal1).Distance(self.matrixdescriptors.new_mol.GetConformer().GetAtomPosition(metal2))
                        if distance > 0 and distance <= 3.5:  # typical metallophilic interaction distance cutoff
                            self.metallophilic_edges.append([metal1, metal2])

    def FindPiPiStacking(self):
        self.pi_pi_edges = []
        for ring1 in self.ring_atoms:
            if all(self.matrixdescriptors.element[atom] in ['C', 'N', 'O', 'S'] for atom in ring1):  # check if all atoms in the ring are aromatic
                for ring2 in self.ring_atoms:
                    if ring1 != ring2 and all(self.matrixdescriptors.element[atom] in ['C', 'N', 'O', 'S'] for atom in ring2):
                        distances = []
                        for atom1 in ring1:
                            for atom2 in ring2:
                                distance = self.matrixdescriptors.new_mol.GetConformer().GetAtomPosition(atom1).Distance(self.matrixdescriptors.new_mol.GetConformer().GetAtomPosition(atom2))
                                distances.append(distance)
                        min_distance = min(distances)
                        if min_distance > 0 and min_distance <= 5:  # typical π-π stacking distance cutoff
                            self.pi_pi_edges.append((ring1, ring2, min_distance))
        
        # Sort by distance and take the two most vertical and nearest ones
        self.pi_pi_edges.sort(key=lambda x: x[2])
        self.pi_pi_edges = self.pi_pi_edges[:2]


    def ElectronegativityDifference(self, edge):
        en_atom1 = self.pauling_dict.get(self.matrixdescriptors.element[edge[0]], 0)
        en_atom2 = self.pauling_dict.get(self.matrixdescriptors.element[edge[1]], 0)
        return abs(en_atom1 - en_atom2)

    def GenerateBondFeatureVector(self):
        self.GenerateEdges()
        self.bond_features = []
        edges = [] 
        for ind in range(len(self.edges_u)):
            bond_feature = []
            edge = sorted([self.edges_u[ind],self.edges_v[ind]])
            edges.append(edge)
            bo = self.EncodeBondOrder()
            bond_feature += self.BondinRing(edge)
            bond_feature += self.BondOrder(edge,bo)
            bond_feature += self.BondConjugation(edge,bo)
            bond_feature += self.BondStereochemistry(edge,bo)
            bond_feature += self.BondRotation(edge,bo)

            if self.params.addmissingbonds == 'hbonds':
                bond_feature += [0]
            
            if self.params.addcho:            
                bond_feature += [0]

            if self.params.adddihydrogenbonds:
                bond_feature += [0]

            if self.params.addcationpi:
                bond_feature += [0]

            if self.params.addpipistack:
                bond_feature += [0]

            if self.params.addhalogenbonds:
                bond_feature += [0]
            
            if self.params.addmetallophilic:
                bond_feature += [0]

            if self.params.addelectrostatic:
                bond_feature += [0,0]

            if self.params.addeneg:
                bond_feature += [self.ElectronegativityDifference(edge)]

            self.bond_features.append(bond_feature)
            self.bond_feature_length = len(bond_feature)

        if self.params.addmissingbonds == 'global':  
            for i in range(len(self.matrixdescriptors.element)):
                for j in range(len(self.matrixdescriptors.element)):
                    bond_feature = []
                    if i > j:
                        if [i,j] not in edges:
                            self.edges_u.append(i)
                            self.edges_v.append(j)
                            bo = self.EncodeBondOrder()
                            bond_feature += self.BondinRing(edge)
                            bond_feature += self.BondOrder(edge,bo)
                            bond_feature += self.BondConjugation(edge,bo)
                            bond_feature += self.BondStereochemistry(edge,bo)
                            bond_feature += self.BondRotation(edge,bo)
                            self.bond_features.append(bond_feature)
                            self.bond_feature_length = len(bond_feature)
        
        
        elif self.params.addmissingbonds == 'hbonds':
            self.FindHBonds()
            self.FindCH_OInteractions()
            self.FindDihydrogenBonds()
            self.FindCationPiInteractions()
            self.FindHalogenBonds()
            self.FindElectrostaticInteractions()
            self.FindMetallophilicInteractions()
            self.FindPiPiStacking()
            for i in range(len(self.matrixdescriptors.element)):
                for j in range(len(self.matrixdescriptors.element)):
                    bond_feature = []
                    if [i,j] not in edges:
                        self.edges_u.append(i)
                        self.edges_v.append(j)
                        bo = self.EncodeBondOrder()
                        bond_feature += self.BondinRing(edge)
                        bond_feature += self.BondOrder(edge,bo)
                        bond_feature += self.BondConjugation(edge,bo)
                        bond_feature += self.BondStereochemistry(edge,bo)
                        bond_feature += self.BondRotation(edge,bo)

                        if [i,j] in self.hbond_edges:
                            bond_feature += [1]
                        else:
                            bond_feature += [0]

                        if self.params.addcho:
                            if [i,j] in self.ch_o_edges:
                                bond_feature += [1]
                            else:
                                bond_feature += [0]

                        if self.params.adddihydrogenbonds:
                            if [i,j] in self.dihydrogen_edges:
                                bond_feature += [1]
                            else:
                                bond_feature += [0]

                        if self.params.addcationpi:
                            if [i,j] in self.cation_pi_edges:
                                bond_feature += [1]
                            else:
                                bond_feature += [0]

                        if self.params.addpipistack:
                            if [i,j] in self.pi_pi_edges:
                                bond_feature += [1]
                            else:
                                bond_feature += [0]

                        if self.params.addhalogenbonds:
                            if [i,j] in self.halogen_bond_edges:
                                bond_feature += [1]
                            else:
                                bond_feature += [0]
                        
                        if self.params.addmetallophilic:
                            if [i,j] in self.metallophilic_edges:
                                bond_feature += [1]
                            else:
                                bond_feature += [0]

                        if self.params.addelectrostatic:
                            if [i,j] in self.electrostatic_edges['attract']:
                                bond_feature += [0,1]
                            elif [i,j] in self.electrostatic_edges['repel']:
                                bond_feature += [1,0]
                            else:
                                bond_feature += [0,0]

                        if self.params.addeneg:
                            bond_feature += [self.ElectronegativityDifference(edge)]                        
                        self.bond_features.append(bond_feature)
                        self.bond_feature_length = len(bond_feature)

    def run(self):
        self.InitializeAddons()
        self.GenerateAtomFeatureVector()
        self.GenerateBondFeatureVector()


class ReactionFeaturizerwithPaddingandGeometry:
    def __init__(self,reaction_smiles,arguments,denotation='>>'):
        self.reaction = reaction_smiles
        self.SplitandCheckForMapping(denotation)
        self.reactant = ReactionComponentFeaturizerwithPaddingandGeometry(self.reactant,arguments)
        self.product = ReactionComponentFeaturizerwithPaddingandGeometry(self.product,arguments)
        self.reactant.run()
        self.product.run()
        self.params = arguments


    def SplitandCheckForMapping(self,denotation):
        self.reactant = self.reaction.split(denotation)[0]
        self.product = self.reaction.split(denotation)[1]

        if not self.IsAtomMapped(self.reactant) or not self.IsAtomMapped(self.product):
            r = self.StripAtomMapping(self.reactant)
            p = self.StripAtomMapping(self.product)

            mapfunction = getattr(self,f'Mapvia{self.params.mappingfunction}',None)
            self.reactant,self.product = mapfunction(r,p)
            

    def MapviaRXNMapper(self,r,p):
        rxn_mapper = RXNMapper()
        rxn = [r + '>>'+p]
        results = rxn_mapper.get_attention_guided_atom_maps(rxn)
        # Get Mapped Reaction
        results = results[0]['mapped_rxn'].split('>>')
        rsmi = results[0]
        psmi = results[1]

        #Add Hydrogens and Map Them
        self.mapper_condfidence = results[0]['confidence'] 
        if self.params.rxnmapper.totalmapping:
            rmol = self.AddHMapping(rsmi)
            pmol = self.AddHMapping(psmi)
            rsmi = Chem.MolToSmiles(rmol)
            psmi = Chem.MolToSmiles(pmol)
        return rsmi,psmi
    
    def MapviaRDKit(self):
        pass

    def MapviaOB3D(self):
        pass

    def MapviaOB2D(self):
        pass


    def AddHMapping(smi):
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        for atom in mol.GetAtoms():
            if atom.GetSymbol() in ['H','Cl','F','I','Br']:
                atom.SetAtomMapNum(atom.GetIdx() + 1)  # Increment the atom mapping label for H atoms
        return Chem.MolToSmiles(mol)

    def StripAtomMapping(self, smi):
        molecule = Chem.MolFromSmiles(smi)
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(molecule)

    def IsAtomMapped(self,smi):
        molecule = Chem.MolFromSmiles(smi)
        for atom in molecule.GetAtoms():
            if atom.GetAtomMapNum() != 0:
                return True
        return False

    def SetFeaturesToNone(self):
        self.reactant.atom_features = None
        self.product.atom_features = None
        self.reactant.bond_features = None
        self.product.bond_features = None
            
    def CheckElementConsistency(self):
        if self.reactant.matrixdescriptors.element != self.product.matrixdescriptors.element and len(self.reactant.matrixdescriptors.element) == len(self.product.matrixdescriptors.element):
            warnings.warn("Element inconsistency between reactant and product, but the same # of atoms are there.")
            self.SetFeaturesToNone()
            return False
        elif self.reactant.matrixdescriptors.element != self.product.matrixdescriptors.element and len(self.reactant.matrixdescriptors.element) != len(self.product.matrixdescriptors.element):
            warnings.warn("Element inconsistency between reactant and product.")
            self.SetFeaturesToNone()
            return False
        else:
            return False
        
    def ReactiveBondInformation(self):
        self.reactionpropeties = Reactive(self.reactant.matrixdescriptors.element,self.reactant.matrixdescriptors.bond_mat,self.product.matrixdescriptors.bond_mat)
    
    def DistanceFromReactingAtom(self,ind,gs):
        if not self.params.removereactiveinfo:
            if len(self.reactionpropeties.atom.reactive) > 0:
                dis = min([gs[ind][indr] for indr in self.reactive_atoms])
            else:
                dis = 0 
            return [dis]
        else:
            return []
        
    def BondChangeInfo(self,edge):
        BO_R = self.reactant.matrixdescriptors.bond_mat[edge[0],edge[1]]
        BO_P = self.product.matrixdescriptors.bond_mat[edge[0],edge[1]]
        if BO_R == BO_P:
            RBtype = self.reactant.properties.bond_encode['T1']
            PBtype = self.reactant.properties.bond_encode['T1']
        elif BO_R == 0.0:
            RBtype = self.reactant.properties.bond_encode['T4']
            PBtype = self.reactant.properties.bond_encode['T3']
        elif BO_P == 0.0:
            RBtype = self.reactant.properties.bond_encode['T3']
            PBtype = self.reactant.properties.bond_encode['T4']
        elif BO_R < BO_P and BO_R > 0:
            RBtype = self.reactant.properties.bond_encode['T2']
            PBtype = self.reactant.properties.bond_encode['T2']
        elif BO_R > BO_P and BO_R > 0:
            RBtype = self.reactant.properties.bond_encode['T5']
            PBtype = self.reactant.properties.bond_encode['T5']
        return RBtype,PBtype

    def OldBondChangeInfo(self,edge):
        BO_R = self.reactant.matrixdescriptors.bond_mat[edge[0],edge[1]]
        BO_P = self.product.matrixdescriptors.bond_mat[edge[0],edge[1]]
        if BO_R == BO_P:
            RBtype = self.reactant.properties.old_bond_encode['T1']
            PBtype = self.reactant.properties.old_bond_encode['T1']
        elif BO_R == 0.0:
            RBtype = self.reactant.properties.old_bond_encode['T4']
            PBtype = self.reactant.properties.old_bond_encode['T3']
        elif BO_P == 0.0:
            RBtype = self.reactant.properties.old_bond_encode['T3']
            PBtype = self.reactant.properties.old_bond_encode['T4']
        elif BO_R != BO_P and BO_R > 0:
            RBtype = self.reactant.properties.old_bond_encode['T2']
            PBtype = self.reactant.properties.old_bond_encode['T2']
        return RBtype,PBtype
    
    def DistanceFromReactingBond(self,edge):
        if self.params.adddisttoreactingbonds:
            if len(self.reactionpropeties.bond.reactive) > 0:
                dis = min([self.reactant.gs[edge[0]][rb[0]] + self.reactant.gs[edge[1]][rb[1]] for rb in self.reactionpropeties.bond.reactive])
            else:
                dis = 0
            return [dis]
        else:
            return []
                    
    def NeighboringReactives(self,adj_mat,ind):
        if self.params.addneighboringreactives: 
            reactive_neighbors = 0
            for neighbor in adj_mat[ind]:
                if neighbor in self.reactionpropeties.atom.reactive:
                    reactive_neighbors += 1
            return [reactive_neighbors]
        else:
            return []
    
    def BondDistanceFromReactingBond(self, edge):
        if self.params.adddisttoreactingbonds:
            if len(self.reactionpropeties.bond.reactive) > 0:
                dis = min([self.reactant.gs[edge[0]][rb[0]] + self.reactant.gs[edge[1]][rb[1]] for rb in self.reactionpropeties.bond.reactive])
            else:
                dis = 0
            return [dis]
        else:
            return []
    
    def BondDistanceChange(self, edge):
        if self.params.adddisttoreactingbonds:
            reactant_conf = self.reactant.new_mol.GetConformer()
            product_conf = self.product.new_mol.GetConformer()
            reactant_distance = reactant_conf.GetAtomPosition(edge[0]).Distance(reactant_conf.GetAtomPosition(edge[1]))
            product_distance = product_conf.GetAtomPosition(edge[0]).Distance(product_conf.GetAtomPosition(edge[1]))
            distance_change = product_distance - reactant_distance
            return [distance_change]
        else:
            return []

    def GenerateBondFeatureVector(self):
        for ind in range(len(self.reactant.edges_u)):
            edge = sorted([self.reactant.edges_u[ind],self.reactant.edges_v[ind]])
            if self.params.oldbondencode: RBtype,PBtype = self.OldBondChangeInfo(edge)
            else: RBtype,PBtype = self.BondChangeInfo(edge)    
            self.reactant.bond_features[ind] += RBtype
            self.product.bond_features[ind] += PBtype
            if self.params.addbonddistance: 
                self.reactant.bond_features[ind] += self.BondDistanceFromReactingBond(edge)
                self.product.bond_features[ind] += self.BondDistanceFromReactingBond(edge)
            if self.params.addbonddistancechange:
                self.reactant.bond_features[ind] += self.BondDistanceChange(edge)
                self.product.bond_features[ind] += self.BondDistanceChange(edge)
            
    def GenerateAtomFeatureVector(self):
        for ind,atom_feature in enumerate(self.reactant.atom_features):
            atom_feature += self.DistanceFromReactingAtom(ind,self.reactant.gs)
            if self.params.addneighboringreactives: atom_feature += self.NeighboringReactives(self.reactant.matrixdescriptors.adj_mat,ind)
        
        
        for ind,atom_feature in enumerate(self.product.atom_features):
            atom_feature += self.DistanceFromReactingAtom(ind,self.product.gs)
            atom_feature += self.NeighboringReactives(self.product.matrixdescriptors.adj_mat,ind)
            if self.params.addneighboringreactives: atom_feature += self.NeighboringReactives(self.product.matrixdescriptors.adj_mat,ind)