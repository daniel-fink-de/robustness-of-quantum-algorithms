import cmath
import numpy as np
import scipy as sp
from bqskit.ir.gates import *

"""
This module contains extension methods for BQSKit.
"""


def get_unitary(gate, noise=None):
    """
    Take a non-noisy gate from bqskit and returns a numpy
    array corresponding to the unitary. If a noise scalar is specified,
    a coherent control error will be added, i.e.,
    exp(-i (1+x) H) with H being the generator of the gate and x the noise term.

    Additionally, the norm of the Hermitian is returned.
    """

    hermitian = get_hermitian_operator(gate)
    unitary = sp.linalg.expm(-1.0j * (1.0 + noise) * hermitian)
    norm = np.linalg.norm(hermitian)

    return unitary, norm


def get_hermitian_operator(unitary):
    """
    Given a unitary U as matrix, this function returns the Hermitian generator of this unitary.
    We do that via diagonalizing U = RDR' with D=diag(lambda), lambda=e^(-i theta) being the eigenvalues.
    Based on that, we get the phases theta and reconstruct the Hermitian operator as H = R diag(theta) R'.
    """

    # Perform diagonalizing U = RDR' with D=diag(lambda), lambda=e^(-i theta) being the eigenvalues
    D, R = np.linalg.eig(unitary)

    # Get the phases theta
    thetas = (-1) * np.vectorize(cmath.phase)(D)

    # Construct the Hermitian generator H = R diag(theta) R'
    hermitian = np.matmul(np.matmul(R, np.diag(thetas)), R.conj().T)

    return hermitian


def compute_fidelity(psi, phi):
    """
    Compute the fidelity as F(psi, rho) = |<psi|phi>|.
    This function is only used for testing purposes below.
    """

    fidelity = np.abs(np.matmul(psi.conj().T, phi))

    return fidelity


def draw_random_state(n_qubits):
    """
    Generate a random state.
    This function is only used for testing purposes below.
    """

    psi = np.random.rand(2 ** n_qubits) + 1j * np.random.rand(2 ** n_qubits)
    psi /= np.linalg.norm(psi)

    return psi


def test():
    """
    Test if the Hermitian decomposition works successfully.
    """

    gate_set_a = {SXGate(), XGate(), RZGate(), CXGate()}
    gate_set_b = {SXGate(), XGate(), RZGate(), CZGate()}
    gate_set_c = {U1Gate(), U2Gate(), U3Gate(), CXGate()}
    gate_set_d = {PhasedXZGate(), RZGate(), SycamoreGate(), CZGate(), SqrtISwapGate()}
    gate_set_e = {U1qPiGate, U1qPi2Gate, RZGate(), ZZGate()}

    unique_gates = gate_set_a\
        .union(gate_set_c)\
        .union(gate_set_c)\
        .union(gate_set_e)\
        .union(gate_set_d)\
        .union(gate_set_b)

    for gate in unique_gates:
        for i in range(200):
            random_parameters = np.random.rand(gate.num_params)
            random_state = draw_random_state(gate.num_qudits)

            unitary = gate.get_unitary(random_parameters).numpy
            exp_unitary, hermitian_norm = get_unitary(unitary, 0.0)

            real_evolution = np.dot(unitary, random_state)
            exp_evolution = np.dot(exp_unitary, random_state)

            fidelity = np.abs(compute_fidelity(real_evolution, exp_evolution))

            if not np.isclose(fidelity, 1.0, rtol=1.e-12):
                raise ValueError(gate)

    print(""
          "All gates of interest could be decomposed into their Hermitian generator representation "
          "U = exp(-i H) successfully."
          "")


if __name__ == "__main__":
    test()
