from pauli_channel_approximation import generate_report
from subprocess import Popen
for i in range(119, 160):
	generate_report('pickled_controls{}.pkl'.format(i))
