from pauli_channel_approximation import *
import dill
import sys
pca = dill.load(open("pickled_controls{}.pkl".format(sys.argv[1]), 'rb'))
for i in range(10):
     pca.plot_dpn(i-1)


