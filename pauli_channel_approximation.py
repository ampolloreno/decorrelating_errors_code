from GRAPE import GRAPE, control_unitaries, adjoint, average_over_noise
import numpy as np
from scipy.optimize import minimize
import dill
import itertools
import matplotlib.pyplot as plt
from copy import deepcopy
from functools import reduce
import subprocess
import os.path
import multiprocessing

#Note ambient hamiltonian as been changed to a list of hamiltonians, but the name remains unchanged.

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

    :param numpy.ndarry sop: Representation our operator as a superoperator, e.g. operating on vectorized density
     matrices.
    """
    basis = PAULIS
    for _ in range(int(np.log2(sop.shape[0]) / 2.0) - 1):
        basis = [np.kron(base, pauli) for pauli in PAULIS for base in basis]
    off_diagonal = 0
    for i, base1 in enumerate(basis):
        for j, base2 in enumerate(basis):
            if i != j:
                off_diagonal += np.abs(np.trace(adjoint(np.kron(base1, base2)).dot(sop))) ** 2
    return off_diagonal


def error_on_controls(ambient_hamiltonian, control_hamiltonians, controls, dt,
                      target_operator):
    unitary = reduce(lambda a, b: a.dot(b),
                     control_unitaries(ambient_hamiltonian, control_hamiltonians,
                                       controls, dt))
    error = error_unitary(unitary, target_operator)
    return error


class PCA(object):
    """Class to perform Pauli Channel Approximations- i.e. pick out weights for families of controls that make them look
     most like Pauli Channels."""

    def __init__(self, num_controls, ambient_hamiltonian, control_hamiltonians, target_operator,
                 num_steps, time, threshold, detunings):
        import time as t
        start = t.time()
        controlset = []
        dt = time / num_steps
        for i in range(num_controls):
            print("CONTROL {}".format(i))
            result = GRAPE(ambient_hamiltonian, control_hamiltonians, target_operator,
                           num_steps, time, threshold, detunings)
            controlset.append(result.reshape(-1, len(control_hamiltonians)))
            # controlset.append(np.array([1] * num_steps).reshape(-1, 1))

        # Initialize random probabilities
        probs = np.random.rand(1, num_controls)
        probs /= np.sum(probs)
        # Enforce bounds on probs
        constraint = (0, 1)
        disp = True
        options = {"disp": disp}
        constraints = {"type": "eq", "fun": lambda x: 1 - sum(x)}

        def off_diagonal_error(probs, controlset, ambient_hamiltonian, control_hamiltonians,
                               detunings, dt, target_operator):
            func = error_on_controls
            avg_errors = []
            for controls in controlset:
                avg_error = average_over_noise(func, ambient_hamiltonian, control_hamiltonians,
                                               controls, detunings, dt, target_operator)
                avg_errors.append(np.array(avg_error))
            avg_error = reduce(lambda a, b: a + b,
                               [avg_errors[i] * prob for i, prob in enumerate(probs)])
            return off_diagonal_projection(avg_error)

        func = lambda x: off_diagonal_error(x, controlset, ambient_hamiltonian, control_hamiltonians, detunings, dt,
                                            target_operator)
        # probs = minimize(func, probs, method='SLSQP', bounds=[constraint for _ in probs[0]], constraints=constraints, options=options, iprint=2)
        import scipy
        probs = scipy.optimize.fmin_slsqp(func, probs, eqcons=[lambda x: 1 - sum(probs)],
                                          bounds=[constraint for _ in probs[0]],
                                          iprint=10)
        self.probs = probs
        self.controlset = controlset

        self.detunings = detunings
        self.target_operator = target_operator
        self.dt = dt
        self.ambient_hamiltonian = ambient_hamiltonian
        self.control_hamiltonians = control_hamiltonians
        stop = t.time()
        self.time = stop - start
        # self.plot_control_fidelity(-1)
        # self.plot_dpn(-1)

    def plot_everything(self, num_processors=63, num_points=3):
        """Plots the depolarizing noise and gate fidelity over all detunings, varying over the list
         provided by itertools."""

        values_to_plot = []
        corr = []
        for i, detuning in enumerate(self.detunings):
            # TODO make params
            values = np.linspace(-3 * detuning[0], 3 * detuning[0], num_points)
            values_to_plot.append(values)
            corr.append(i)
        combinations = itertools.product(*values_to_plot)
        new_combinations = []
        for combo in combinations:
            new_combo = []
            for index in corr:
                new_combo.append(combo[index])
            new_combinations.append(new_combo)
        combinations = new_combinations
        pool = multiprocessing.Pool(num_processors)
        lst = [(self.controlset, self.ambient_hamiltonian, combo, self.dt,
                self.control_hamiltonians, self.target_operator, self.probs)
               for combo in combinations]
        projs_fidelities = pool.map(compute_dpn_and_fid, lst)
        pool.close()
        projs = [pf[0] for pf in projs_fidelities]
        fidelities = [pf[1] for pf in projs_fidelities]

        # projs2 = []
        # for proj in projs:
        #     from numbers import Number
        #     if not isinstance(proj, Number):
        #         projs2.append(proj)
        #
        # projs = projs2
        projs = np.vstack(projs).T
        fidelities = np.vstack(fidelities).T
        plt.figure(1,  figsize=(16, 8))  # the first figure
        plt.subplot(211)  # the first subplot in the first figure
        for i, row in enumerate(projs[:-1, :]):
            plt.plot(range(len(row)), row)
        plt.plot(range(len(projs[-1, :])), projs[-1, :], label="min", color='k', linewidth=2, zorder=10)
        plt.legend()
        plt.semilogy()


        plt.subplot(212)  # the second subplot in the first figure
        for i, row in enumerate(fidelities[:-1, :]):
            plt.plot(range(len(row)), row)
        plt.plot(range(len(fidelities[-1, :])), fidelities[-1, :], label="min", color='k', linewidth=2, zorder=10)
        plt.legend()
        plt.tight_layout()


def compute_dpn_and_fid(data):
    controlset, ambient_hamiltonian0, combo, dt, control_hamiltonians, target_operator, probs = data
    print("DOING COMBO {}".format(combo))
    fidelities = []
    projs = []
    sops = []
    controlset_unitaries = []
    #
    # for i, com in enumerate(combo):
    #     if i != 0 and com != 0:
    #         return 0
    for controls in controlset:
        newcontrols = deepcopy(controls)
        ambient_hamiltonian = [deepcopy(ah).astype("float") for ah in ambient_hamiltonian0]
        for cnum, value in enumerate(combo):
            cnum -= len(ambient_hamiltonian0)
            if cnum >= 0:
                newcontrols[:, cnum] = newcontrols[:, cnum] * (1 + value)
            if cnum < 0:
                ambient_hamiltonian[cnum] *= float(value)
        step_unitaries = control_unitaries(ambient_hamiltonian,
                                           control_hamiltonians, newcontrols,
                                           dt)
        unitary = reduce(lambda a, b: a.dot(b), step_unitaries)
        sop = error_unitary(unitary, target_operator)
        sops.append(sop)
        projs.append(off_diagonal_projection(sop))
        controlset_unitaries.append(unitary)
        fidelity = np.trace(adjoint(target_operator).dot(unitary))/target_operator.shape[0]
        fidelity *= np.conj(fidelity)
        fidelities.append(fidelity)
    avg_sop = reduce(lambda a, b: a + b, [prob * sops[i] for i, prob in enumerate(probs)])

    balanced = reduce(lambda a, b: a + b,
                      [probs[i] * np.kron(np.conj(unitary), unitary) for i, unitary in
                       enumerate(controlset_unitaries)])
    balanced_fidelity = np.trace(
        adjoint(balanced).dot(np.kron(np.conj(target_operator), target_operator)))/(target_operator.shape[0])**2

    fidelities.append(balanced_fidelity)
    projs.append(off_diagonal_projection(avg_sop))

    fidelities = np.array(fidelities).T
    projs = np.array(projs).T

    return projs, fidelities

# #Deprecated
#     def plot_control_fidelity(self, cnum):
#         # Vary over ith parameter
#         if self.detunings[cnum + 1] == 0:
#             return
#         values = np.linspace(-self.detunings[cnum + 1] * 3, self.detunings[cnum + 1] * 3, 25)
#         control_fidelities = []
#         for value in values:
#             fidelities = []
#             controlset_unitaries = []
#             for controls in self.controlset:
#                 newcontrols = deepcopy(controls)
#                 ambient_hamiltonian = deepcopy(self.ambient_hamiltonian).astype("float")
#                 if cnum >= 0:
#                     newcontrols[:, cnum] = newcontrols[:, cnum] * (1 + value)
#                 if cnum == -1:
#                     ambient_hamiltonian *= float(value)
#                 step_unitaries = control_unitaries(ambient_hamiltonian, self.control_hamiltonians, newcontrols, self.dt)
#                 unitary = reduce(lambda a, b: a.dot(b), step_unitaries)
#                 fidelity = np.trace(adjoint(self.target_operator).dot(unitary))
#                 fidelity *= np.conj(fidelity)
#                 fidelities.append(fidelity)
#                 controlset_unitaries.append(unitary)
#             balanced = reduce(lambda a, b: a + b, [self.probs[i] * np.kron(np.conj(unitary), unitary) for i, unitary in
#                                                    enumerate(controlset_unitaries)])
#             balanced_fidelity = np.trace(
#                 adjoint(balanced).dot(np.kron(np.conj(self.target_operator), self.target_operator)))
#             fidelities.append(-balanced_fidelity)
#             control_fidelities.append(fidelities)
#         control_fidelities = np.array(control_fidelities).T
#         for i, row in enumerate(control_fidelities[:-1]):
#             plt.plot(values, row)
#         plt.xlabel("Detuning for {}th controllable parameter in the Hamiltonian with mean 1.\n (-1 is uncontrollable, mean 0)".format(cnum))
#         plt.ylabel("Fidelity of the control.")
#         plt.title("Control Fidelity versus Detuning for the {}th control".format(cnum))
#         plt.legend()
#         plt.plot(values, -control_fidelities[-1], label="min", color='k', linewidth=2)
#         plt.tight_layout()
#         # import os
#         # i = 0
#         # while os.path.exists("image%s.png" % i):
#         #     i += 1
#         # plt.savefig("image%s.png" % i)
#         # plt.clf()
#
# # Deprecated
#     def plot_dpn(self, cnum, num_processors=7):
#         # # Assume we vary the first free parameter
#         # if self.detunings[cnum + 1] == 0:
#         #     return
#         basis = PAULIS
#         dim = self.target_operator.shape[0]
#         for _ in range(int(np.log2(dim) - 1)):
#             basis = [np.kron(base, pauli) for pauli in PAULIS for base in basis]
#         values = np.linspace(-self.detunings[cnum + 1], self.detunings[cnum + 1], 3)
#         control_fidelities = []
#         for value in values:
#             fidelities = []
#             controlset_unitaries = []
#             sops = []
#             for controls in self.controlset:
#                 newcontrols = deepcopy(controls)
#                 ambient_hamiltonian = deepcopy(self.ambient_hamiltonian).astype("float")
#                 if cnum >= 0:
#                     newcontrols[:, cnum] = newcontrols[:, cnum] * (1 + value)
#                 if cnum == -1:
#                     ambient_hamiltonian *= float(value)
#                 step_unitaries = control_unitaries(ambient_hamiltonian, self.control_hamiltonians, newcontrols, self.dt)
#                 unitary = reduce(lambda a, b: a.dot(b), step_unitaries)
#                 sop = error_unitary(unitary, self.target_operator)
#                 sops.append(sop)
#                 fidelities.append(off_diagonal_projection(sop))
#                 controlset_unitaries.append(unitary)
#             sop = reduce(lambda a, b: a + b, [prob * sops[i] for i, prob in enumerate(self.probs)])
#             fidelities.append(off_diagonal_projection(sop))
#             control_fidelities.append(fidelities)
#         control_fidelities = np.array(control_fidelities).T
#         for i, row in enumerate(control_fidelities[:-1, :]):
#             plt.plot(values, row)
#         plt.xlabel("Detuning for {}th controllable parameter in the Hamiltonian with mean 1.\n (-1 is uncontrollable, mean 0)".format(cnum))
#         plt.ylabel("Projection onto off diagonal elements of the PTM")
#         plt.title("Off diagonal contribution versus detuning for the {}th control".format(cnum))
#         plt.plot(values, control_fidelities[-1, :], label="min", color='k', linewidth=2)
#         plt.legend()
#         plt.semilogy()
#         plt.tight_layout()
#         # print self.probs
#         # import os
#         # i = 0
#         # while os.path.exists("image%s.png" % i):
#         #     i += 1
#         # plt.savefig("image%s.png" % i)
#         # plt.clf()
#         #
#


def load_pca(filename):
    try:
        with open(filename, 'rb') as fileh:
            pca = dill.load(fileh)
    except ValueError:
        subprocess.Popen(["python3", "repickle.py", filename])
        import time
        time.sleep(2)
        with open("python2" + filename, 'r') as fileh:
            data = fileh.read()
            pca = dill.loads(data)
    return pca


def generate_report(filename):
    import subprocess
    import time
    report_dir = "report_{}".format(filename.split('.')[0])
    if os.path.isfile(report_dir + "/report.tex"):
        print("We already made this one, moving on!")
        return
    subprocess.Popen(["mkdir", report_dir])
    pca = load_pca(filename)
    try:
        ambient_hamiltonian = pca.ambient_hamiltonian
        control_hamiltonians = pca.control_hamiltonians
        target_operator = pca.target_operator
        detunings = pca.detunings
        dt = pca.dt
        probs = pca.probs
        time = pca.time
        controlset = pca.controlset
    except AttributeError:
        pass
    image_locs = []
    pca.plot_everything()
    image_locs.append(report_dir + "/control_dpn_all.png")
    plt.savefig(report_dir + "/control_dpn_all.png")
    plt.clf()
    #
    # for i in range(len(pca.control_hamiltonians) + 1):
    #     pca.plot_control_fidelity(i - 1)
    #     control_fid_loc = report_dir + "/control_fid_{}".format(i)
    #     image_locs.append("control_fid_{}".format(i))
    #     plt.savefig(control_fid_loc)
    #     plt.clf()
    #     pca.plot_dpn(i - 1)
    #     off_diag_loc = report_dir + "/off_diag_{}".format(i)
    #     image_locs.append("off_diag_{}".format(i))
    #     plt.savefig(off_diag_loc)
    #     plt.clf()

    latex = r"""
\documentclass{article}
\usepackage{graphicx}
\usepackage[margin=1mm]{geometry}
\graphicspath{ {images/} }
     
\begin{document}
    """
    latex += "\n\n\\newpage"
    latex += "\n\nAmbient Hamiltonian: " + str(ambient_hamiltonian)
    latex += "\n\nControl Hamiltonians:" + str(control_hamiltonians)
    latex += "\n\nDetunings: " + str(detunings)
    latex += "\n\n dt: " + str(dt)
    latex += "\n\nProbs: " + str(probs)
    latex += "\n\nTarget Operator: " + str(target_operator)
    latex += "\n\nTime: " + str(time)
    latex += "\n\nControlset: " + str(controlset)
    latex += "\n\\begin{center}"
    for image_loc in image_locs:
        latex += """\n\\includegraphics[scale=.9]{{{}}}""".format(image_loc)
    latex += "\n\n\end{center}"
    latex += "\n" + r"\end{document}"
    with open(report_dir + "/report.tex", 'w') as fileh:
        fileh.write(latex)


def generate_all_reports():
    import os
    for filename in os.listdir(os.getcwd()):
        if filename.split('.')[-1] == "pkl" and  "aws" in filename.split('.')[0]:
            generate_report(filename)


if __name__ == "__main__":
    np.random.seed(1000)
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1.j], [1.j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    #CNOT = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    IZ = np.kron(I, Z)
    ZI = np.kron(Z, I)
    XI = np.kron(X, I)
    IX = np.kron(I, X)
    IY = np.kron(I, Y)
    YI = np.kron(Y, I)
    ZZ = np.kron(Z, Z)
    entangle_ZZ = np.array([[1, 0, 0, 0], [0, -1.j, 0, 0], [0, 0, -1.j, 0], [0, 0, 0, 1]])
    # applied multiplicatively
    ambient_hamiltonian = [IZ, ZI]
    control_hamiltonians = [IX, IY, XI, YI, ZZ]
    detunings = [(.001, 1), (.001, 1), (.001, 2), (.001, 2), (.001, 1)]
    target_operator = entangle_ZZ
    time = 2 * np.pi
    num_steps = 200
    threshold = 1 - .001
    num_controls = 100
    pca = PCA(num_controls, ambient_hamiltonian, control_hamiltonians, target_operator,
              num_steps, time, threshold, detunings)
    print("TOOK {}".format(pca.time))
    import os
    i = 0
    while os.path.exists("pickled_controls%s.pkl" % i):
        i += 1
    fh = open("pickled_controls%s.pkl" % i, "wb")
    dill.dump(pca, fh)
    fh.close()

####################################################################################################
# if __name__ == "__main__":
    # np.random.seed(1000)
    # I = np.eye(2)
    # II = np.kron(I, I)
    # X = np.array([[0, 1], [1, 0]])
    # IX = np.kron(I, X)
    # XI = np.kron(X, I)
    # Y = np.array([[0, -1.j], [1.j, 0]])
    # IY = np.kron(I, Y)
    # YI = np.kron(Y, I)
    # Z = np.array([[1, 0], [0, -1]])
    # IZ = np.kron(I, Z)
    # ZI = np.kron(Z, I)
    # XXYY = np.kron(X, X) + np.kron(Y, Y)
    # ISWAP = np.array([[1, 0, 0, 0], [0, 0, -1.j, 0], [0, -1.j, 0, 0], [0, 0, 0, 1]])
    # # H = (Z + X) / np.sqrt(2)
    # # applied multiplicatively
    # unisup = 1 / 2.0 * np.array([1, 1, 1, 1])
    # unisup = 2 * np.outer(unisup, unisup)
    # unisup -= np.eye(4)
    # assert np.isclose(adjoint(unisup).dot(unisup), np.eye(4)).all()
    #
    # ambient_hamiltonian = IZ
    # control_hamiltonians = [IX, XI, IY, YI, IZ, ZI, XXYY]
    # target_operator = unisup
    # time = 2 * np.pi
    # num_steps = 1000
    # threshold = 1 - .001
    # num_controls = 100
    # pca = PCA(num_controls, ambient_hamiltonian, control_hamiltonians, target_operator, num_steps, time,
    #           threshold, [.001] + [.001, .001, .001, .001, 0, 0, .001])
    # print("TOOK {}".format(pca.time))
    # import os
    #
    # i = 0
    # while os.path.exists("pickled_controlsaws%s.pkl" % i):
    #     i += 1
    # fh = open("pickled_controlsaws%s.pkl" % i, "wb")
    # dill.dump(pca, fh)
    # fh.close()
    #

# if __name__ == "__main__":
#     np.random.seed(1000)
#     I = np.eye(2)
#     X = np.array([[0, 1], [1, 0]])
#     Y = np.array([[0, -1.j], [1.j, 0]])
#     Z = np.array([[1, 0], [0, -1]])
#     H = (Z + X)/np.sqrt(2)
#     # applied multiplicatively
#     ambient_hamiltonian = Z
#     control_hamiltonians = [X, Y, Z]
#     target_operator = H
#     time = 4 * np.pi
#     num_steps = 50
#     threshold = 1 - .001
#     num_controls = 5
#     pca = PCA(num_controls, ambient_hamiltonian, control_hamiltonians, target_operator, num_steps, time, threshold,
#               [.001] + [.001 for _ in control_hamiltonians])
#     print("TOOK {}".format(pca.time))
#     import os
#     i = 0
#     while os.path.exists("pickled_controls%s.pkl" % i):
#         i += 1
#     fh = open("pickled_controls%s.pkl" % i, "wb")
#     dill.dump(pca, fh)
#     fh.close()

    # if __name__ == "__main__":
    #     np.random.seed(1000)
    #     I = np.eye(2)
    #     X = np.array([[0, 1], [1, 0]])
    #     Y = np.array([[0, -1.j], [1.j, 0]])
    #     Z = np.array([[1, 0], [0, -1]])
    #     H = (Z + X)/np.sqrt(2)
    #     ambient_hamiltonian = .01 * Z
    #     control_hamiltonians = [X, Y, Z]
    #     target_operator = H
    #     time = 6 * np.pi
    #     num_steps = 500
    #     threshold = 1 - .001
    #     num_controls = 10
    #     pca = PCA(num_controls, ambient_hamiltonian, control_hamiltonians, target_operator, num_steps, time, threshold,
    #               [.001] + [.01 for _ in control_hamiltonians])
    #     print "TOOK {}".format(pca.time)
    #     import os
    #     i = 0
    #     while os.path.exists("pickled_controls%s.pkl" % i):
    #         i += 1
    #     fh = open("pickled_controls%s.pkl" % i, "wb")
    #     dill.dump(pca, fh)
    #     fh.close()
