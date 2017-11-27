from pauli_channel_approximation import *
from subprocess import Popen
for i in [250]:
	generate_report('pickled_controls{}.pkl'.format(i))
