from pauli_channel_approximation import *
import dill
import sys
pca = dill.load(open("pickled_controls{}.pkl".format(sys.argv[1]), 'rb'))
pca.plot_dpn(-1)
pca.plot_dpn(0)
pca.plot_dpn(1)
pca.plot_dpn(2)


