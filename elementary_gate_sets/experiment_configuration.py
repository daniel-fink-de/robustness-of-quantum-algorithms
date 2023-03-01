from dataclasses import dataclass
from pickle import dump, load
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from numpy import abs, matmul, dot
from numpy.random import uniform
from qiskit.circuit.library import QFT
from bqskit.ir.gates import *
from bqskit.qis import UnitaryBuilder, UnitaryMatrix
from bqskit.ext.rigetti import rigetti_gate_set
from bqskit.ext.honeywell import honeywell_gate_set
from bqskit.ext.cirq.models import google_gate_set
from bqskit.ext.qiskit import qiskit_to_bqskit
from qiskit.quantum_info import random_statevector
from bqskit_extensions import get_unitary


"""
This module contains global experiment configurations.
"""


class Parameters:
    """
    Stores the parameters for the entire experiment.
    """
    N_QUBITS = 3
    EPSILON = 0.05
    N_INSTANCES = 40_000
    OPTIMIZATION_LEVEL = 3
    MAX_SYNTHESIS_SIZE = 3
    SYNTHESIS_EPSILON = 1e-12
    GATE_SET_NAMES = ["a", "b", "c", "d", "e"]


@dataclass
class Experiment:
    gate_set_name: str
    fidelities: NDArray
    bound: float
    n_gates: int
    depth: int


class Gates:
    @staticmethod
    def A():
        """
        The elementary gate set used by current IBM devices.
        """

        return {SXGate(), XGate(), RZGate(), CXGate()}

    @staticmethod
    def B():
        """
        The elementary gate set used by Rigetti devices.
        """

        return rigetti_gate_set

    @staticmethod
    def C():
        """
        The elementary gate set used by old IBM devices.
        """

        return {U1Gate(), U2Gate(), U3Gate(), CXGate()}

    @staticmethod
    def D():
        """
        The elementary gate set used by Google devices.
        """

        return google_gate_set

    @staticmethod
    def E():
        """
        The elementary gate set used by Honeywell devices.
        """

        return honeywell_gate_set


class Circuits:
    @staticmethod
    def perfect():
        """
        Return a Qiskit circuit that corresponds to the QFT.
        """

        circuit = QuantumCircuit(Parameters.N_QUBITS)

        # Add the QFT
        qft = QFT(Parameters.N_QUBITS)
        circuit = circuit.compose(qft)

        return circuit

    @staticmethod
    def to_noisy_unitary(circuit, epsilon):
        """
        Takes the given BQSKit circuit and converts it to the noisy unitary,
        that is, each gate will be changed to its coherent control error form
        where noise is drawn uniformly at random with the draw_noise function.
        Additionally, the sum of the Hermitian norms is returned.
        """

        builder = UnitaryBuilder(circuit.num_qudits, circuit.radixes)
        Lipschitz_bound = 0.0

        for operation in circuit:
            unitary = operation.get_unitary()

            noise = Functions.draw_noise(epsilon, 1)
            unitary_noisy, hermitian_norm = get_unitary(unitary, noise=noise)
            unitary_matrix = UnitaryMatrix(unitary_noisy)
            Lipschitz_bound = Lipschitz_bound + hermitian_norm

            builder.apply_right(unitary_matrix, operation.location)

        unitary_circuit = builder.get_unitary()

        return unitary_circuit, Lipschitz_bound

    @staticmethod
    def perfect_state(psi):
        """
        Returns the perfect output state phi given an input state psi as numpy array.
        """

        qc = Circuits.perfect()
        circuit = qiskit_to_bqskit(qc)
        unitary = circuit.get_unitary()
        phi = dot(unitary, psi)

        return phi


class Functions:
    @staticmethod
    def write_object_to_file(obj, name):
        dump(obj, open(f"data/{name}.p", "wb"))
        return

    @staticmethod
    def read_object_from_file(name):
        obj = load(open(f"data/{name}.p", "rb"))
        return obj

    @staticmethod
    def draw_noise(epsilon, dim):
        """
        Draw dim random variables uniformly at random from [-epsilon, +epsilon]
        """
        noise = uniform(-epsilon, +epsilon, dim)

        return noise

    @staticmethod
    def compute_fidelity(psi, phi):
        """
        Compute the fidelity as F(psi, rho) = |<psi|phi>|.
        """

        fidelity = abs(matmul(psi.conj().T, phi))

        return fidelity

    @staticmethod
    def draw_random_state(n_qubits):
        """
        Draw a state uniformly at random (Haar).
        """
        psi = random_statevector(2 ** n_qubits)

        return psi

    @staticmethod
    def save_circuit_to_file(circuit, name):
        """
        Save the given bqskit circuit to a qasm file.
        """

        circuit.save(f"data/{name}.qasm")

        return
