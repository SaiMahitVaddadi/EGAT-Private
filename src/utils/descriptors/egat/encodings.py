from .properties import Properties

class Encodings(Properties):
    def __init__(self):
        """
        Initializes the Encodings class and sets up various encoding schemes.
        """
        super().__init__()
        self.element()
        self.chirality()
        self.hybridization()
        self.bond()
        self.bondorder()
        self.bondstereo()
        self.bondrotation()
        self.hybridization_full()
        self.old_bond_encoding()

    def element(self):
        """
        Sets up element encoding.
        self.element_encode: dict
            Key: str (element name)
            Value: list [int (atomic number), float (atomic mass)]
        """
        self.element_encode = {key.capitalize(): [value, self.el_mass[key.capitalize()]] for key, value in self.el_to_an.items()}
    
    def chirality(self):
        """
        Sets up chirality encoding.
        self.atom_chiral_encode: dict
            Key: str (chirality type)
            Value: list [int, int, int]
        """
        self.atom_chiral_encode = {'?': [0,0,1], 'R': [0,1,0], 'S': [1,0,0]}

    def hybridization(self):
        """
        Sets up hybridization encoding.
        self.atom_hybrid_encode: dict
            Key: Chem.HybridizationType
            Value: list [int, int, int, int]
        """
        self.atom_hybrid_encode = {Chem.HybridizationType.S: [0,0,0,1], Chem.HybridizationType.SP: [0,0,1,0], Chem.HybridizationType.SP2: [0,1,0,0], 
                                   Chem.HybridizationType.SP3: [1,0,0,0]}

    def hybridization_full(self):
        """
        Sets up full hybridization encoding.
        self.atom_hybrid_encode: dict
            Key: Chem.HybridizationType
            Value: list [int, int, int, int, int, int, int, int, int]
        """
        self.atom_hybrid_encode = {Chem.HybridizationType.S: [0,0,0,1,0,0,0,0,0], Chem.HybridizationType.SP: [0,0,1,0,0,0,0,0,0], Chem.HybridizationType.SP2: [0,1,0,0,0,0,0,0,0], 
                                   Chem.HybridizationType.SP3: [1,0,0,0,0,0,0,0,0],Chem.HybridizationType.SP3D: [0,0,0,0,1,0,0,0,0],Chem.HybridizationType.SP3D2: [0,0,0,0,0,1,0,0,0],
                                   Chem.HybridizationType.OTHER: [0,0,0,0,0,0,1,0,0],Chem.HybridizationType.UNSPECIFIED: [0,0,0,0,0,0,0,1,0],Chem.HybridizationType.SP2D: [0,0,0,0,0,0,0,0,1]}

    def bond(self):
        """
        Sets up bond encoding.
        self.bond_encode: dict
            Key: str (bond type)
            Value: list [int, int, int, int, int]
        """
        self.bond_encode = {'T1': [0,0,0,1,0],'T2':[0,0,1,0,0],'T3':[0,1,0,0,0],'T4':[1,0,0,0,0],'T5':[0,0,0,0,1]}

    def old_bond_encoding(self):
        """
        Sets up old bond encoding.
        self.old_bond_encode: dict
            Key: str (bond type)
            Value: list [int, int, int, int]
        """
        self.old_bond_encode = {'T1': [0,0,0,1],'T2':[0,0,1,0],'T3':[0,1,0,0],'T4':[1,0,0,0]}
    
    def bondorder(self):
        """
        Sets up bond order encoding.
        self.bond_order_encode: dict
            Key: str (bond order type)
            Value: list [int, int, int, int, int]
        """
        self.bond_order_encode = {'B0': [0,0,0,0,1],'B1':[0,0,0,1,0],'B2':[0,0,1,0,0],'B3':[0,1,0,0,0],'BA':[1,0,0,0,0]}
    
    def bondstereo(self):
        """
        Sets up bond stereo encoding.
        self.bond_stereo_encode: dict
            Key: str (stereo type)
            Value: list [int, int, int]
        """
        self.bond_stereo_encode = {'ANY': [0,0,1], 'E': [0,1,0], 'Z': [1,0,0]}
    
    def bondrotation(self):
        """
        Sets up bond rotation encoding.
        self.bond_rotat_encode: dict
            Key: str (rotation type)
            Value: list [int, int]
        """
        self.bond_rotat_encode = {'TRUE':[1,0],'FALSE':[0,1]}
    
    def hydrogenbond(self):
        """
        Sets up hydrogen bond encoding.
        self.h_bond_encode: dict
            Key: str (hydrogen bond presence)
            Value: list [int, int]
        """
        self.h_bond_encode = {'TRUE':[1,0],'FALSE':[0,1]}
    
    def globalbond(self):
        """
        Sets up global bond encoding.
        self.global_bond_encode: dict
            Key: str (global bond presence)
            Value: list [int, int]
        """
        self.global_bond_encode = {'TRUE':[1,0],'FALSE':[0,1]}
    
    def functionalgroup(self):
        """
        Sets up functional group encoding.
        self.functional_group_encode: dict
            Key: str (functional group presence)
            Value: list [int, int]
        """
        self.functional_group_encode = {'TRUE':[1,0],'FALSE':[0,1]}