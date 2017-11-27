from pauli_channel_approximation import *
from subprocess import Popen
for i in [251, 252]:
	generate_report('pickled_controls{}.pkl'.format(i))
