from GRAPE import GRAPE, control_unitaries, adjoint, average_over_noise
import numpy as np
from scipy.optimize import minimize
import dill
import matplotlib.pyplot as plt
from copy import deepcopy


I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1.j], [1.j, 0]])
Z = np.array([[1, 0], [0, -1]])
PAULIS = [I, X, Y, Z]


def error_unitary(unitary, target):
    """
    Constructs the superoperator for the error unitary corresponding propagating forward by target,
     and back by unitary.

    :param np.array unitary: Control unitary realized in the computational basis.
    :param np.array target: The desired unitary in the computational basis.
    :return: The superoperator corresponding to the unitary postrotational error.
    :rtype: numpy.array
    """
    error = adjoint(unitary).dot(target)
    return np.kron(np.conj(error), error)


def off_diagonal_projection(sop):
    """
    Computes a quantity proportional to the sum of the magnitudes of terms corresponding to non-stochastic evolution.

    :param probs:
    :param controlset:
    :param ambient_hamiltonian:
    :param control_hamiltonians:
    :param detunings:
    :param dim:
    :param dt:
    :param target_operator:
    :return:
    """
    basis = PAULIS
    for _ in xrange(int(np.log2(sop.shape[0]) - 2)):
        basis = [np.kron(base, pauli) for pauli in PAULIS for base in basis]
    off_diagonal = 0
    for i, base1 in enumerate(basis):
        for j, base2 in enumerate(basis):
            if i != j:
                off_diagonal += np.abs(np.trace(adjoint(np.kron(base1, base2)).dot(sop)))**2
    return off_diagonal


class PCA(object):
    def __init__(self, num_controls, ambient_hamiltonian, control_hamiltonians, target_operator,
                 num_steps, time, threshold, detunings):
        controlset = []
        np.random.seed(123)
        dt = time/num_steps
        for _ in xrange(num_controls):
            result = GRAPE(ambient_hamiltonian, control_hamiltonians, target_operator,
                                  num_steps, time, threshold, detunings)
            controlset.append(result.reshape(-1, len(control_hamiltonians)))
        # Initialize random probabilities
        probs = np.random.rand(1, num_controls)
        probs /= np.sum(probs)
        # Enforce bounds on probs
        constraint = (0, 1)
        disp = True
        options = {"disp": disp}
        constraints = {"type": "eq", "fun": lambda probs: 1 - sum(probs)}

        def off_diagonal_error(probs, controlset, ambient_hamiltonian, control_hamiltonians,
                               detunings, dt, target_operator):
            def error_on_controls(ambient_hamiltonian, control_hamiltonians, controls, dt,
                                  target_operator):
                unitary = reduce(lambda a, b: a.dot(b),
                                 control_unitaries(ambient_hamiltonian, control_hamiltonians,
                                                   controls, dt))
                error = error_unitary(unitary, target_operator)
                return error

            func = error_on_controls
            avg_errors = []
            for controls in controlset:
                avg_error = average_over_noise(func, ambient_hamiltonian, control_hamiltonians,
                                               controls, detunings, dt, target_operator)
                avg_errors.append(np.array(avg_error))
            avg_error = reduce(lambda a, b: a + b,
                               [avg_errors[i] * prob for i, prob in enumerate(probs)])
            return off_diagonal_projection(avg_error)

        func = lambda x: off_diagonal_error(x, controlset, ambient_hamiltonian, control_hamiltonians, detunings, dt, target_operator)
        probs = minimize(func, probs, method='SLSQP', bounds=[constraint for _ in probs[0]], constraints=constraints, options=options)

        self.probs = probs.x
        self.controlset = controlset
        self.detunings = detunings
        self.target_operator = target_operator
        self.dt = dt
        self.ambient_hamiltonian = ambient_hamiltonian
        self.control_hamiltonians = control_hamiltonians
        self.plot_control_fidelity(0)
        self.plot_dpn(0)

    def plot_control_fidelity(self, cnum):
        # Vary over ith parameter
        values = np.arange(-self.detunings[cnum+1], self.detunings[cnum+1], self.detunings[cnum+1]/50.0)
        control_fidelities = []
        for value in values:
            fidelities = []
            controlset_unitaries = []
            for controls in self.controlset:
                newcontrols = controls
                newcontrols[cnum, :] = newcontrols[cnum, :] * (1 + value)
                step_unitaries = control_unitaries(self.ambient_hamiltonian, self.control_hamiltonians, controls, self.dt)
                unitary = reduce(lambda a, b: a.dot(b), step_unitaries)
                fidelity = np.trace(adjoint(self.target_operator).dot(unitary))
                fidelity *= np.conj(fidelity)
                fidelities.append(fidelity)
                controlset_unitaries.append(unitary)
            balanced = reduce(lambda a, b: a + b, [self.probs[i] * np.kron(np.conj(unitary), unitary) for i, unitary in enumerate(controlset_unitaries)])
            balanced_fidelity = np.trace(adjoint(balanced).dot(np.kron(np.conj(self.target_operator), self.target_operator)))
            fidelities.append(-balanced_fidelity)
            control_fidelities.append(fidelities)
        control_fidelities = np.array(control_fidelities).T
        for i, row in enumerate(control_fidelities[:-1]):
            plt.plot(values, row, label=i)
        plt.plot(values, -control_fidelities[-1], label="min", color='k', linewidth=2)
        plt.legend()
        print self.probs
        plt.show()

    def plot_dpn(self, cnum):
        # Assume we vary the first free parameter
        basis = PAULIS
        dim = self.target_operator.shape[0]
        for _ in xrange(int(np.log2(dim) - 1)):
            basis = [np.kron(base, pauli) for pauli in PAULIS for base in basis]
        values = np.arange(-self.detunings[cnum+1], self.detunings[cnum+1], self.detunings[cnum+1]/50.0)
        control_fidelities = []
        for value in values:
            fidelities = []
            controlset_unitaries = []
            sops = []
            for controls in self.controlset:
                newcontrols = deepcopy(controls)
                newcontrols[:, cnum] = newcontrols[:, cnum] * (1 + value)
                step_unitaries = control_unitaries(self.ambient_hamiltonian, self.control_hamiltonians, newcontrols, self.dt)
                unitary = reduce(lambda a, b: a.dot(b), step_unitaries)
                sop = error_unitary(unitary, self.target_operator)
                sops.append(sop)
                fidelities.append(off_diagonal_projection(sop))
                controlset_unitaries.append(unitary)
            sop = reduce(lambda a, b: a + b, [prob * sops[i] for i, prob in enumerate(self.probs)])
            fidelities.append(off_diagonal_projection(sop))
            control_fidelities.append(fidelities)
        control_fidelities = np.array(control_fidelities).T
        for i, row in enumerate(control_fidelities[:-1, :]):
            plt.plot(values, row, label=i)
        plt.plot(values, control_fidelities[-1, :], label="min", color='k', linewidth=2)
        plt.legend()
        print self.probs
        plt.show()

if __name__ == "__main__":
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1.j], [1.j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    ambient_hamiltonian = 0 * I
    control_hamiltonians = [X]
    target_operator = X
    time = 7 * np.pi
    num_steps = 10
    threshold = 1 - 1E-3
    num_controls = 2

    pca = PCA(num_controls, ambient_hamiltonian, control_hamiltonians, target_operator, num_steps, time, threshold,
        [.001 for _ in control_hamiltonians] + [.001])
    dill.dump(pca, open('pickled_controls', 'wb'))