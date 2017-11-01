from mpi4py import MPI
import time

COMM = MPI.COMM_WORLD


def f():
    def split(container, count):
        """
	    Simple function splitting a container into equal length chunks.
	    Order is not preserved but this is potentially an advantage depending on
	    the use case.
	    """
        return [container[_i::count] for _i in range(count)]
    if COMM.rank == 0:
        jobs = [1 for _ in range(7)]
        print(jobs)
        # Split into however many cores are available.
        jobs = split(jobs, COMM.size)
    else:
        jobs = None
    print(jobs)
    return
    # Scatter jobs across cores.
    jobs = COMM.scatter(jobs, root=0)
    # Now each rank just does its jobs and collects everything in a results list.
    # Make sure to not use super big objects in there as they will be pickled to be
    # exchanged over MPI.
    results = []
    for job in jobs:
        print("SLEEPING!", time.time())
        time.sleep(job)
        results.append(1)
        print(time.time())

    # Gather results on rank 0.
    results = MPI.COMM_WORLD.gather(results, root=0)
    if COMM.rank == 0:
        # Flatten list of lists.
        results = [_i for temp in results for _i in temp]
    results = COMM.rank
    return results


def g():
    return f()


print(g())
