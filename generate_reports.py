from pauli_channel_approximation import *
from subprocess import Popen
for i in [255]:
	generate_report('pickled_controls{}.pkl'.format(i))
