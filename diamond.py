"""
diamond.py

Compute the diamond norm of a quantum process

Kevin Young
Sandia National Laboratories
2 September 2015

------------------------------------------------------

Uses the primal SDP from arXiv:1207.5726v2, Sec 3.2

Maximize 1/2 ( < J(phi), X > + < J(phi).dag, X.dag > )
Subject to  [[ I otimes rho0, X],
            [X.dag, I otimes rho1]] >> 0
              rho0, rho1 are density matrices
              X is linear operator

Jamiolkowski representation of the process
  J(phi) = sum_ij Phi(Eij) otimes Eij

< A, B > = Tr(A.dag B)
"""

from scipy import transpose, array, zeros
import scipy
from itertools import chain, repeat
import cvxpy as cvx
import pickle
import numpy as np

def vec(matrix_in):
    # Stack the columns of a matrix to return a vector
    return [b for a in transpose(matrix_in) for b in a]

def unvec(vector_in):
    # Slice a vector into columns of a matrix
    dim = int(scipy.sqrt(len(vector_in)))
    return transpose(array(list(zip(*[chain(vector_in, repeat(None, dim-1))]*dim))))

def jamiolkowski(process, representation = 'superoperator'):
    # Return the Choi-Jamiolkowski representation of a quantum process
    # Add methods as necessary to accept different representations
    process = array(process)
    if representation == 'unitary':
        process = np.kron(process.conj(), process)
        representation = 'superoperator'
    if representation == 'superoperator':
        # Superoperator is the linear operator acting on vec(rho)
        dimension = int(scipy.sqrt(process.shape[0]))
        jamiolkowski_matrix = zeros([dimension**2, dimension**2], dtype='complex')
        for i in range(dimension**2):
            Ei_vec= zeros(dimension**2)
            Ei_vec[i] = 1
            output = unvec(scipy.dot(process,Ei_vec))
            jamiolkowski_matrix += scipy.kron(output, unvec(Ei_vec))
        return jamiolkowski_matrix

def diamond_norm( jamiolkowski_matrix ):
    dim = jamiolkowski_matrix.shape[0]
    udim = int(scipy.sqrt(dim))

    # Here we define a bunch of auxiliary matrices because CVXPY doesn't use complex numbers

    K = jamiolkowski_matrix.real # J.real
    L = jamiolkowski_matrix.imag # J.imag

    Y = cvx.Variable(dim, dim) # X.real
    Z = cvx.Variable(dim, dim) # X.imag

    sig0 = cvx.Variable(udim,udim) # rho0.real
    sig1 = cvx.Variable(udim,udim) # rho1.real
    tau0 = cvx.Variable(udim,udim) # rho0.imag
    tau1 = cvx.Variable(udim,udim) # rho1.imag

    ident = scipy.eye(udim)

    objective = cvx.Maximize( cvx.trace( K.T * Y + L.T * Z) )
    constraints = [ cvx.bmat( [
                        [ cvx.kron(ident, sig0), Y, -cvx.kron(ident, tau0), -Z],
                        [ Y.T, cvx.kron(ident, sig1), Z.T, -cvx.kron(ident, tau1)],
                        [ cvx.kron(ident, tau0), Z, cvx.kron(ident, sig0), Y],
                        [ -Z.T, cvx.kron(ident, tau1), Y.T, cvx.kron(ident, sig1)]] ) >> 0,
                    cvx.bmat( [[sig0, -tau0],
                           [tau0,  sig0]] ) >> 0,
                    cvx.bmat( [[sig1, -tau1],
                           [tau1,  sig1]] ) >> 0,
                    sig0 == sig0.T,
                    sig1 == sig1.T,
                    tau0 == -tau0.T,
                    tau1 == -tau1.T,
                    cvx.trace(sig0) == 1.,
                    cvx.trace(sig1) == 1. ]

    prob = cvx.Problem(objective, constraints)
    prob.solve(solver="CVXOPT")

    return prob.value


