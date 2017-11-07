import os
import dill
import sys
from pauli_channel_approximation import generate_report, PCA
from mpi4py import MPI
COMM = MPI.COMM_WORLD
# for i,filename in enumerate(os.listdir()):
import time
for i in reversed(range(160)):
	time.sleep(3)
	filename = 'pickled_controls{}.pkl'.format(i)
	# if filename == 'pickled_controls66.pkl':
	# 	continue
	print(i, filename)
	print(COMM.rank)
	sys.stdout.flush()
	try:
		print(COMM.rank)
		sys.stdout.flush()
		if len(filename) >=4 and filename[-4:] == '.pkl':
			print(COMM.rank)
			sys.stdout.flush()
			if COMM.rank == 0:
				fileh = open(filename, 'rb')
				pca = dill.load(fileh)
				fileh.close()
			else:
				pca = None
			# harcode 4 for now, num cores
			pca = COMM.bcast(pca, root=0)
			PCA.assign_probs(pca)
			if COMM.rank == 0:
				fileh = open(filename, 'wb')
				dill.dump(pca, fileh)
				fileh.close()
				# generate_report(filename)
	except KeyError as e:
		print(e)