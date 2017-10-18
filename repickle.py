import dill
import sys
import pickle
from pauli_channel_approximation import PCA
arg = sys.argv[1]
with open(arg, "rb") as f:
    w = dill.load(f)
with open("python2" + arg, 'wb') as fileh:
    pickle.dump(w, fileh, protocol=1)

