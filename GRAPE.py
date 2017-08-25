from __future__ import division

import itertools
import numpy as np
import scipy
import scipy.optimize as optimize
import matplotlib.pyplot as plt
from numpy.polynomial.hermite import hermgauss
import multiprocessing



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
        evolution = scipy.linalg.expm(-1.j * dt * (ambient_hamiltonian + np.sum(step_hamiltonian, axis=0)))
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


def average_over_noise(func, ambient_hamiltonian, control_hamiltonians,
                       controls, detunings, dt, target_operator, deg=2):
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
    points, weights = hermgauss(deg)
    pairs = [zip(detuning * points, weights) for i, detuning in enumerate(detunings)]
    controls = controls.reshape(-1, len(control_hamiltonians))
    average_perf = 0
    combinations = itertools.product(*pairs)
    pool = multiprocessing.Pool(7)
 
    for combination in combinations:
        new_controls = [[control * (1 + combination[i+1][0]) for i, control in enumerate(row)]
                        for row in controls]
        new_controls = np.array(new_controls).flatten()
        average_perf += reduce(lambda a, b: a * b, [comb[1] for comb in combination]) * \
                        func(ambient_hamiltonian * (combination[0][0]), control_hamiltonians, new_controls, dt,
                             target_operator) / reduce(lambda a, b: a * b, detunings)
    return average_perf


def GRAPE(ambient_hamiltonian, control_hamiltonians, target_operator, num_steps, time,
          threshold=1-1E-3, detunings=None):
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
    dt = time/num_steps
    #deg = 1
    if detunings is not None:
        perf = lambda controls: average_over_noise(grape_perf, ambient_hamiltonian, control_hamiltonians, controls, detunings, dt, target_operator)#, deg=deg)
        grad = lambda controls: average_over_noise(grape_gradient, ambient_hamiltonian, control_hamiltonians, controls, detunings, dt, target_operator)#, deg=deg)
    else:
        perf = lambda controls: grape_perf(ambient_hamiltonian, control_hamiltonians, controls, dt,
                                           target_operator)
        grad = lambda controls: grape_gradient(ambient_hamiltonian, control_hamiltonians, controls,
                                               dt, target_operator)
    dimension = np.shape(ambient_hamiltonian)[0]
    disp = True
    ftol = (1-threshold)
    options = {"ftol": ftol,
               "disp": disp}
    constraint = (-1, 1)
    controls = (2.0 * np.random.rand(1, int(len(control_hamiltonians) * num_steps)) - 1.0) * .1
    result = optimize.minimize(fun=perf, x0=controls, jac=grad, method='tnc', options=options)
                               #bounds=[constraint for _ in controls[0]],

    # Verify that the controls meet requirements at zero.
    perf_at_zero = grape_perf(ambient_hamiltonian * 0,
                              control_hamiltonians,
                              result.x, dt,
                              target_operator)
    print "PERFORMANCE IS: ", (-perf_at_zero)/dimension**2
    while (-perf_at_zero)/dimension**2 < threshold:
        print "PERFORMANCE IS: ",  (-perf_at_zero)/dimension**2
        print "RETRYING GRAPE FOR BETTER CONTROLS"
        controls = (2.0 * np.random.rand(1, int(len(control_hamiltonians) * num_steps)) - 1.0) * .1
        result = optimize.minimize(fun=perf, x0=controls, jac=grad, method='tnc', options=options)
                                   #bounds=[constraint for _ in controls[0]], options=options)
        print "minimize finished, performance is  {}".format(-result.fun/dimension**2)
        perf_at_zero = grape_perf(ambient_hamiltonian * 0,
                                  control_hamiltonians,
                                  result.x, dt,
                                  target_operator)
    return result.x

if __name__ == "__main__":
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1.j], [1.j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    ambient_hamiltonian = Z
    control_hamiltonians = [X, Y, Z]
    target_operator = (X + Z)/np.sqrt(2)
    assert np.isclose(target_operator.dot(adjoint(target_operator)), np.eye(target_operator.shape[0])).all()
    time = 2 * np.pi
    num_steps = 200
    x = GRAPE(ambient_hamiltonian, control_hamiltonians, target_operator, num_steps, time, detunings=[.001] * (len(control_hamiltonians) + 1), threshold=.999)
    controls = x.reshape(-1, len(control_hamiltonians))
    print reduce(lambda a, b: a.dot(b), control_unitaries(ambient_hamiltonian, control_hamiltonians, controls, time/num_steps))
    plt.step(range(len(controls.flatten())), controls.flatten())
    plt.show()
    from scipy.integrate import ode
    from numpy import real, array, pi, dot, reshape, conjugate

    sigI = [[1., 0], [0, 1.]]
    sigX = [[0, 1.], [1., 0]]
    sigY = [[0, -1.j], [1.j, 0]]
    sigZ = [[1., 0], [0, -1.]]


    def ham(t):
        dt = time/num_steps
        x = controls
        return ambient_hamiltonian * 0 + np.sum([control * control_hamiltonians[i] for i, control in enumerate(x[int(t/dt)])], axis=0)


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
            expectations += [map(lambda op: expect(op, r.y), [sigX, sigY, sigZ])]
        except:
            pass

    from qutip import Bloch

    bloch = Bloch()
    bloch.add_points(real(zip(*expectations)), 'l')
    bloch.show()
