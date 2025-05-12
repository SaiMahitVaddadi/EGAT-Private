import sys
def trace(frame, event, arg):
    print("%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno))
    return trace
sys.path.append("/Users/svaddadi/Documents/GitHub/")
import EGAT
from EGAT import src
from EGAT.src.dataset.base.dataset_setups import DatasetParams
from EGAT.src.params.graph import GraphParams
from EGAT.src.loader.Loader import EGATDataLoader