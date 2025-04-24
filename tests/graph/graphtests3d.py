import sys
sys.path.append("/Users/svaddadi/Documents/GitHub/")
import EGAT
from EGAT import src
from EGAT.src.graph.molecular.MoleculeGeometry import MoleculeFeaturizerwithGeometry
from EGAT.src.params.graph import GraphParams
import json

params = GraphParams()

# This is the way to set the parameters if you want explicit Acid-Base Information
params.acidbase = 'Lewis'
params.getacidbaseinfo = True
# This is the way to get radicals
# Note: For YARP,  YARP can give multiple smiles based on the resonance structure. The way we handle this is by obtaining x different graphs for the same y.
# With that, we would need to change the dataloader to handle such cases as it also happens with the conformational sampling. 
params.getradical = 'YARP'
# This is the way to get spiro and brdgehead information. 
params.getspiro = True
params.getbridgehead = True
# This is the way to get fused ring information.
params.getfusedinformation = True
# This is the way to get rotatable bond information.
params.getrotatablebonds = True
# This is the way to get gasteiger charges.
params.charge = 'Gasteiger'
# This is the way to get BRICS Decomposition locations. 
params.getbrics = True
params.checkbricsbond = True 
# This is the way to get Electronegativity locations. 
params.getelectronegativity = True



#To Get 3D, you need to do this 
params.conformer.generator = 'RDKit'
params.conformer.nconfs = 1

#Add the position 
params.removecoordinationinfo = False
# Add Sterimol Information
params.getsterimol = True
# Check Distance to the Center of Mass in the Molecule
params.getdistancetocenterofmass = True

# 



graph = MoleculeFeaturizerwithGeometry('CCCCC',params)
graph.run()

def dump_to_json(obj, filename):
    attributes = {attr: getattr(obj, attr) for attr in dir(obj) if not callable(getattr(obj, attr)) and not attr.startswith("__")}
    with open(filename, 'w') as json_file:
        json.dump(attributes, json_file, default=str, indent=4)

# Example usage:
dump_to_json(params, 'graph_params.json')
dump_to_json(graph, 'graph_attributes.json')
