from pauli_channel_approximation import *
import dill
pca = dill.load(open("pickled_controls", 'rb'))
pca.plot_dpn(0)
