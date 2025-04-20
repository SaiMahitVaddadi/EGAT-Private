from ..base import BaseFeaturizer



class NonBondedInformation(BaseFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
    
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
    
    
    
    def DefaultNBInteraction(self,edge):
        bond_feature = []
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
        return bond_feature
    
