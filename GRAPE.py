import itertools
import numpy as np
import scipy
import scipy.optimize as optimize
import matplotlib.pyplot as plt
from numpy.polynomial.hermite import hermgauss
import multiprocessing
from functools import reduce
from mpi4py import MPI

COMM = MPI.COMM_WORLD


def split(container, count):
    """
    Simple function splitting a container into equal length chunks.
    Order is not preserved but this is potentially an advantage depending on
    the use case.
    """
    return [container[_i::count] for _i in range(count)]


def adjoint(operator):
    """
    Computes the adjoint of a given matrix.

    :param numpy.array operator: Array to compute the adjoint of.
    :return: The adjoint of operator.
    :rtype: numpy.array
    """
    return np.conj(operator).T


def control_unitaries(ambient_hamiltonian, control_hamiltonians, controls, dt):
    """
    Given an controllable and a non-controllable hamiltonian, with controls and the time step size,
    computes the unitaries that give the time evolution. e.g.

    control_unitaries(I, [X], [[0],[1],[0],[1]], np.pi)
    [I, X, I, X]

    :param numpy.array ambient_hamiltonian: A square 2D array, corresponding to the uncontrollable
     part of the system hamiltonian.
    :param list control_hamiltonians: A list of square 2D numpy.arrays, corresponding to the
     controllable terms in the hamiltonian. This should be as long as the second dimension of
     controls.
    :param numpy.array controls: A 2D numpy.array, whose rows represent time steps, and whose
     jth column corresponds to the control amplitude for the jth element of control_hamiltonians.
    :param float dt: The time to evolve each row of controls for.
    :return: The list of unitaries corresponding to the discretized evolution of the system.
    :rtype: list
    """
    if len(controls.shape) == 1:
        controls = controls.reshape(-1, len(control_hamiltonians))
    unitaries = []
    for row in controls:
        step_hamiltonian = [control * control_hamiltonians[i] for i, control in enumerate(row)]
        evolution = scipy.linalg.expm(
            -1.j * dt * (np.sum(ambient_hamiltonian, axis=0) + np.sum(step_hamiltonian, axis=0)))
        unitaries.append(evolution)
    return unitaries


def grape_perf(ambient_hamiltonian, control_hamiltonians, controls, dt, target_operator):
    """
    Evaluate the performance function :math: `\phi_4` as described in Khaneja et al.'s paper.

    :param numpy.array ambient_hamiltonian: A square 2D array, corresponding to the uncontrollable
     part of the system hamiltonian.
    :param list control_hamiltonians: A list of square 2D numpy.arrays, corresponding to the
     controllable terms in the hamiltonian. This should be as long as the second dimension of
     controls.
    :param numpy.array controls: A 1D numpy.array, a flattened version of a 2D.array,
     whose rows represent time steps, and whose jth column corresponds to the control amplitude
     for the jth element of control_hamiltonians.
    :param float dt: The time to evolve each row of controls for.
    :param numpy.array target_operator: The operator we are trying to approximate, given as a
     numpy.array.
    :return: The real valued evaluation of the performance function.
    :rtype: float
    """
    controls = controls.reshape((-1, len(control_hamiltonians)))
    unitaries = control_unitaries(ambient_hamiltonian, control_hamiltonians, controls, dt)
    final_unitary = reduce(np.dot, list(reversed(unitaries)), np.eye(unitaries[0].shape[0]))
    overlap = np.trace(adjoint(target_operator).dot(final_unitary))
    return -overlap * np.conj(overlap)


def grape_gradient(ambient_hamiltonian, control_hamiltonians, controls, dt, target_operator):
    """
    Evaluate the gradient of the performance function :math: `\phi_4` as described in
     Khaneja et al.'s paper.

    :param numpy.array ambient_hamiltonian: A square 2D array, corresponding to the uncontrollable
     part of the system hamiltonian.
    :param list control_hamiltonians: A list of square 2D numpy.arrays, corresponding to the
     controllable terms in the hamiltonian. This should be as long as the second dimension of
     controls.
    :param numpy.array controls: A 1D numpy.array, a flattened version of a 2D.array,
     whose rows represent time steps, and whose jth column corresponds to the control amplitude
     for the jth element of control_hamiltonians.
    :param float dt: The time to evolve each row of controls for.
    :param numpy.array target_operator: The operator we are trying to approximate, given as a
     numpy.array.
    :return: The real valued evaluation of the gradient of the performance function with respect to
     controls. This is flattened to work with scipy.minimize.
    :rtype: numpy.arrays
    """
    controls = controls.reshape((-1, len(control_hamiltonians)))
    unitaries = control_unitaries(ambient_hamiltonian, control_hamiltonians, controls, dt)
    forward = [unitaries[0]]
    backward = [target_operator]
    for unitary in unitaries[1:]:
        forward.append(forward[-1].dot(unitary))
    for unitary in list(reversed(unitaries))[:-1]:
        backward.append(adjoint(unitary).dot(backward[-1]))
    backward = list(reversed(backward))
    grad = np.zeros(controls.shape)
    for i, row in enumerate(grad):
        for j, col in enumerate(row):
            overlap = np.trace(adjoint(forward[i]).dot(backward[i]))
            diff = np.trace(adjoint(backward[i]).dot(control_hamiltonians[j].dot(forward[i])))
            grad[i][j] = 2.0 * np.real(overlap * diff * 1.j * dt)
    return grad.flatten()


def comp_avg_perf(pair):
    combination, controls, func, ambient_hamiltonian, control_hamiltonians, detunings, dt, target_operator = pair
    new_controls = [[control * (1 + combination[i + len(ambient_hamiltonian)][0]) for i, control in
                     enumerate(row)]
                    for row in controls]
    new_controls = np.array(new_controls).flatten()
    nonzero_detunings = np.array(detunings)[np.where(np.array(detunings) != 0)[0]]
    # if func == grape_perf:
    #     print("VALUE {}".format(func([ambient * combination[i][0] for i, ambient in enumerate(ambient_hamiltonian)], control_hamiltonians, new_controls, dt,
    #                      target_operator)))
    #     print(combination)
    # average_perf = reduce(lambda a, b: a * b, [comb[1] for comb in combination]) * \
    #                 func([ambient * combination[i][0] for i, ambient in enumerate(ambient_hamiltonian)], control_hamiltonians, new_controls, dt,
    #                      target_operator) / (np.sqrt(np.pi) ** len(nonzero_detunings))
    average_perf = reduce(lambda a, b: a * b, [comb[1] for comb in combination]) * \
                   func([ambient * combination[i][0] for i, ambient in
                         enumerate(ambient_hamiltonian)], control_hamiltonians, new_controls, dt,
                        target_operator) / (np.sqrt(np.pi) ** len(nonzero_detunings))

    return average_perf


def average_over_noise(func, ambient_hamiltonian, control_hamiltonians,
                       controls, detunings, dt, target_operator, deg=2, num_processors=7):
    """
    Average the given func over noise using gaussian quadrature.

    :param function func: Function that needs to be averaged.
    :param numpy.array ambient_hamiltonian: The uncontrolled Hamiltonian.
    :param list control_hamiltonians: The controllable Hamiltonians.
    :param numpy.array controls: 2D array of controls, of dimension (num_steps, num_controls)
    :param list detunings: Standard deviations of the gaussian distributions over the control knobs.
     The first element should be the detuning on the ambient_hamiltonian
    :param float dt: The time per time step.
    :param numpy.array target_operator: The operator trying to be approximated.
    :param int deg: The degree of polynomial that quadrature will work on.
    :return: The average of func over detunings, scaled by some factor. (TODO Need to make sure the quadrature
     coefficients are being handled correctly)
    :rtype: rtype of func
    """
    controls = controls.reshape(-1, len(control_hamiltonians))

    if COMM.rank == 0:
        corr = []
        if type(detunings[0]) != tuple:
            pass
        else:
            new_detunings = []
            for i, detune in enumerate(detunings):
                new_detunings.append(detune[0])
                for _ in range(detune[1]):
                    corr.append(i)  # use the ith detuning more than once
            detunings = new_detunings
        points, weights = hermgauss(deg)
        nonzero_detunings = np.where(np.array(detunings) != 0)[0]
        zero_detunings = np.where(np.array(detunings) == 0)[0]
        # print([np.sqrt(detuning) * points for i, detuning in enumerate(np.array(detunings)[nonzero_detunings])])
        # print([1/np.sqrt(detuning) * points for i, detuning in enumerate(np.array(detunings)[nonzero_detunings])])

        pairs = [list(zip(np.sqrt(detuning) * points, weights)) for i, detuning in
                 enumerate(np.array(detunings)[nonzero_detunings])]
        for index in zero_detunings:
            pairs.insert(index, [(0, 1)])
        combinations = itertools.product(*pairs)
        # Expand them if there are correlations
        if corr:
            new_combinations = []
            for combo in combinations:
                new_combo = []
                last_number = -1
                for index in corr:
                    if index == last_number:
                        new_number = False
                    else:
                        new_number = True
                    last_number = index
                    pair = combo[index]
                    if not new_number:
                        pair = (pair[0], 1)
                    new_combo.append(pair)
                new_combinations.append(new_combo)
            combinations = new_combinations
        # pool = multiprocessing.Pool(num_processors)
        #
        lst = [(combination, controls, func, ambient_hamiltonian, control_hamiltonians, detunings,
                dt, target_operator) for
               combination in combinations]
        # results = pool.map(comp_avg_perf, lst)
        # pool.close()
        jobs = combinations
        # Split into however many cores are available.
        jobs = split(jobs, COMM.size)
    else:
        jobs = None

    # Scatter jobs across cores.
    jobs = COMM.scatter(jobs, root=0)
    # Now each rank just does its jobs and collects everything in a results list.
    # Make sure to not use super big objects in there as they will be pickled to be
    # exchanged over MPI.
    results = []
    for job in jobs:
        # print("{} has {} jobs, doing job {}".format(COMM.rank, len(jobs), job[0]))
        results.append(comp_avg_perf((
                                     job, controls, func, ambient_hamiltonian, control_hamiltonians,
                                     detunings, dt, target_operator)))
    # Gather results on rank 0.
    results = MPI.COMM_WORLD.allgather(results)

    # if COMM.rank == 0:
    #     # Flatten list of lists.
    results = [_i for temp in results for _i in temp]
    # if np.sum(results, axis=0).shape == ():
    #     print(np.sum(results, axis=0))
    return np.sum(results, axis=0)


def GRAPE(ambient_hamiltonian, control_hamiltonians, target_operator, num_steps, time,
          threshold=1 - 1E-3, detunings=None):
    """
    Perform the GRAPE algorithm to approximate target_gate in num_steps with time given by time,
    using ambient_hamiltonian as the uncontrolled hamiltonian and control_hamiltonians as the
    controllable hamiltonians.

    :param numpy.array ambient_hamiltonian: A square 2D array, corresponding to the uncontrollable
    part of the system hamiltonian.
    :param list control_hamiltonians: A list of square 2D numpy.arrays, corresponding to the
    controllable terms in the hamiltonian. This should be as long as the second dimension of
    controls.
    :param numpy.array target_operator: The operator we are trying to approximate, given as a
    numpy.array.
    :param int num_steps: The number of time steps to give GRAPE to try to approximate
     target_operator. This does NOT refer to the number of steps in the optimization routine.
    :param float time: The total time over which the system is to be evolving. time/num_steps
     is thus the time alloted for each step in the controlled, discretized evolution.
    :param float threshold: The minimum performance of the controls that this function will return.
    :param list detunings: A list of floats corresponding to the standard deviation of the
     uncertainty in the controls. None by default. If specified, there should be one value for each
     element in control_hamiltonians.
    :return: A numpy.array of controls that approximate target_operator with at least threshold
     performance.
    :rtype: numpy.array
    """
    dt = time / num_steps
    # deg = 1
    if detunings is not None:
        perf = lambda controls: average_over_noise(grape_perf, ambient_hamiltonian,
                                                   control_hamiltonians, controls, detunings, dt,
                                                   target_operator)  # , deg=deg)
        grad = lambda controls: average_over_noise(grape_gradient, ambient_hamiltonian,
                                                   control_hamiltonians, controls, detunings, dt,
                                                   target_operator)  # , deg=deg)
    else:
        perf = lambda controls: grape_perf(ambient_hamiltonian, control_hamiltonians, controls, dt,
                                           target_operator)
        grad = lambda controls: grape_gradient(ambient_hamiltonian, control_hamiltonians, controls,
                                               dt, target_operator)
    dimension = np.shape(ambient_hamiltonian[0])[0]
    disp = True
    ftol = (1 - threshold)
    options = {"ftol": ftol,
               "disp": disp}


    # num_samples = 10
    # # We'll assume one control
    # fwhm = 2.5
    # ts = np.arange(num_samples)
    # sigma = 0.5 * fwhm / np.sqrt(2.0 * np.log(2.0))
    # vals = np.exp(-0.5 * (ts - 5) ** 2 / sigma ** 2)
    # epsilon = 1

    vals = [0.050226199732051155, 0.13199339787014291, 0.23855965632270518, 0.34398290898284473, 0.41077865874536323, 0.41077865874536329, 0.34398290898284478, 0.23855965632270532, 0.13199339787014291, 0.050226199732051197]


    constraint = (min(vals), 1)
    #
    controls = np.reshape(vals, (1, len(vals)))
               # + (np.random.rand(1, len(vals)) - .5)*epsilon \
               # + (np.random.rand(1) - .6) * .1
    # (np.random.rand(1) - .4) * .1

    #controls = (2.0 * np.random.rand(1, int(len(control_hamiltonians) * num_steps)) - 1.0)
    # pi_pulse = np.random.randint(2)
    # num_pi_steps = round(np.pi / dt)
    # print(num_pi_steps)
    import sys
    sys.stdout.flush()
    bounds = [constraint for _ in controls[0]]
    # for i in range(num_pi_steps):
    #     print("PI_PULSE", pi_pulse)
    #     import sys
    #     sys.stdout.flush()
    #     if pi_pulse != -2:
    #         controls[0][i*len(control_hamiltonians)] = pi_pulse
    #         controls[0][i*len(control_hamiltonians) + 1] = 0
    #         bounds[i*len(control_hamiltonians)] = (pi_pulse, pi_pulse)
    #         bounds[i*len(control_hamiltonians) + 1] = (0, 0)
    # Start with a pi pulse

    # for i in range(len(controls)):
    #     if np.random.randint(2):
    #         controls[i] = 0


    result = optimize.minimize(fun=perf, x0=controls, jac=grad, method='tnc', options=options,
                               bounds=bounds)

    # Verify that the controls meet requirements at zero.
    perf_at_zero = grape_perf(np.array(ambient_hamiltonian) * 0,
                              control_hamiltonians,
                              result.x, dt,
                              target_operator)

    print("PERF AT ZERO: {}".format(perf_at_zero))

    #    check_perf = perf(result.x)
    print("PERFORMANCE IS: ", (-perf_at_zero) / dimension ** 2)
    import sys
    sys.stdout.flush()
    while (-perf_at_zero) / dimension ** 2 < threshold:
        print("RETRYING GRAPE FOR BETTER CONTROLS")
        sys.stdout.flush()
        controls = (2.0 * np.random.rand(1, int(len(control_hamiltonians) * num_steps)) - 1.0) * .1
        result = optimize.minimize(fun=perf, x0=controls, jac=grad, method='tnc', options=options,
                                   bounds=bounds)
        # bounds=[constraint for _ in controls[0]], options=options)
        print("minimize finished, performance is  {}".format(-result.fun / dimension ** 2))
        perf_at_zero = grape_perf(ambient_hamiltonian * 0,
                                  control_hamiltonians,
                                  result.x, dt,
                                  target_operator)
        # check_perf = perf(result.x)
        print("PERFORMANCE IS: ", (-perf_at_zero) / dimension ** 2)
        sys.stdout.flush()
    return result.x


if __name__ == "__main__":
    np.random.seed(100)
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1.j], [1.j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    ambient_hamiltonian = [I]
    control_hamiltonians = [X, Z]
    target_operator = X
    assert np.isclose(target_operator.dot(adjoint(target_operator)),
                      np.eye(target_operator.shape[0])).all()
    time = 2 * np.pi
    num_steps = 20
    x = GRAPE(ambient_hamiltonian, control_hamiltonians, target_operator, num_steps, time,
              detunings=[.0001] * (len(control_hamiltonians) + len(ambient_hamiltonian)),
              threshold=.9)
    controls = x.reshape(-1, len(control_hamiltonians))
    print(reduce(lambda a, b: a.dot(b),
                 control_unitaries(ambient_hamiltonian, control_hamiltonians, controls,
                                   time / num_steps)))
    plt.step(list(range(len(controls.flatten()))), controls.flatten())
    plt.show()
    from scipy.integrate import ode
    from numpy import real, array, pi, dot, reshape, conjugate

    sigI = [[1., 0], [0, 1.]]
    sigX = [[0, 1.], [1., 0]]
    sigY = [[0, -1.j], [1.j, 0]]
    sigZ = [[1., 0], [0, -1.]]


    def ham(t):
        dt = time / num_steps
        x = controls
        return np.sum(
            [control * control_hamiltonians[i] for i, control in enumerate(x[int(t / dt)])], axis=0)


    def schrodinger(t, y):
        return -1.j * dot(ham(t), y)


    def jacobian(t, y):
        return -1.j * ham(t)


    def expect(op, psi):
        psi = reshape(psi, [2, 1])
        return dot(dot(conjugate(psi.T), op), psi)[0, 0]


    t0 = 0
    psi0 = array([[1.], [0.]])

    r = ode(schrodinger, jacobian).set_integrator('zvode', method='adams',
                                                  with_jacobian=True).set_initial_value(psi0, t0)
    t1 = time
    dt = 0.0001
    expectations = []
    while r.successful() and r.t < t1:
        try:
            r.integrate(r.t + dt)
            expectations += [[expect(op, r.y) for op in [sigX, sigY, sigZ]]]
        except:
            pass

    from qutip import Bloch

    bloch = Bloch()
    bloch.add_points(real(list(zip(*expectations))), 'l')
    bloch.show()
