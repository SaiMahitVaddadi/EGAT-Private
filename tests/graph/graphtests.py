import sys
sys.path.append("/Users/svaddadi/Documents/GitHub/")
import EGAT
from EGAT import src
from EGAT.src.graph.molecular.Molecule import MoleculeFeaturizer
from EGAT.src.params.graph import GraphParams
import toml
import json

params = GraphParams()

# This is the way to set the parameters if you want explicit Acid-Base Information
params.acidbase = 'Lewis'
params.getacidbaseinfo = True

# This is the way to get radicals
params.getradical = 'RDKit'

# This is the way to get spiro and brdgehead information
params.getspiro = True
params.getbridgehead = True


# This is the way to get fused ring information
params.getfusedinformation = True



graph = MoleculeFeaturizer('CCCCC',params)
graph.run()

def dump_to_json(obj, filename):
    attributes = {attr: getattr(obj, attr) for attr in dir(obj) if not callable(getattr(obj, attr)) and not attr.startswith("__")}
    with open(filename, 'w') as json_file:
        json.dump(attributes, json_file, default=str, indent=4)

# Example usage:
dump_to_json(params, 'graph_params.json')
dump_to_json(graph, 'graph_attributes.json')
