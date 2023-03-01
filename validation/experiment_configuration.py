from qiskit import QuantumCircuit
from numpy import pi, linspace, sqrt, matmul, array, cdouble, append, flip, multiply
from numpy.random import uniform
from qiskit.providers.ibmq import IBMQ
from qiskit_aer import QasmSimulator
from qiskit_aer.noise import NoiseModel
from pickle import dump, load
from dataclasses import dataclass
from json import loads, dumps

"""
This module contains global experiment configurations.
"""


class Parameters:
    """
    Stores the parameters for the entire experiment.

    Let us count how many circuits we need to execute for
    the IBM qpu. We want to run 16 different noise levels
    epsilon, and each 80 instances. That are 16 * 80 = 1280
    circuits. However, we need to perform a quantum state
    tomography to calculate the fidelity. Thus, we must run
    each circuit 3 times to collect the 3 Pauli expectation
    values, i.e., we run 3 * 1280 = 3840 circuits. That is
    only for one circuit (e.g., the one with smaller Lipschitz
    bound). Therefore, we have 2 * 3840  = 7680 circuits
    (for smaller and larger Lipschitz constant).
    Each circuit is executed 20,000 times, that are 153,600,000
    circuit evaluations in total.

    At IBMQ, we can only run 5 Job Sets at the same time, with
    each Job Set can consist of 100 circuits. We would like
    to submit jobs homogeneous, such that we always run a full
    batch, but with fewer statistics at a time. Thus, we run
    16 (epsilons) * 3 (expectation values) * 10 (instances)
    = 480 circuits at a time, which can be divided into
    5 job sets. In total, we submit 16x a batch of 5 Job Sets.
    """

    NUMPY_SEED = 513231
    IBM_QPU = "ibm_nairobi"
    IBM_EPSILONS = linspace(0, 1.0, 16)
    IBM_INSTANCES = 10
    IBM_BATCHES = 16
    SIMULATION_EPSILONS = linspace(-1.0, +1.0, 80)
    SIMULATION_INSTANCES = 80
    OPTIMIZATION_LEVEL = 3
    N_SHOTS = 20_000
    L_SMALL = 1 / 8 * pi
    L_LARGE = 5 / 8 * pi
    BOUND = lambda L, epsilon: 1 - (L ** 2 * epsilon ** 2) / 2
    IBM_KEY = ""  # set your IBM Key here


class Circuits:
    @staticmethod
    def perfect():
        """
        This circuit has no Lipschitz bound, since it is without errors.
        """
        qc = QuantumCircuit(1)
        theta = 5 / 4 * pi
        qc.x(0)
        qc.sx(0)
        qc.rz(theta, 0)
        qc.sx(0)
        qc.save_statevector()

        return qc

    @staticmethod
    def perfect_state():
        simulator = Qiskit.get_perfect_simulator()
        job = simulator.run(Circuits.perfect(), shots=1)
        state = job.result().get_statevector()
        psi = state.data

        return psi

    @staticmethod
    def small(qc, noise):
        """
        This circuit has a Lipschitz bound of theta/2 = 1/8 * pi.
        Note, the SX gates are not affected by noise.
        """
        theta = 1 / 4 * pi
        qc.sx(0)
        qc.rz((1 + noise[0]) * theta, 0)
        qc.sx(0)

        return qc

    @staticmethod
    def large(qc, noise):
        """
        This circuit has a Lipschitz bound of theta/2 = 5/8 * pi.
        Note, the SX and X gates are not affected by noise.
        """
        theta = 5 / 4 * pi
        qc.x(0)
        qc.sx(0)
        qc.rz((1 + noise[0]) * theta, 0)
        qc.sx(0)

        return qc


class Functions:
    @staticmethod
    def draw_noise(epsilon, dim):
        """
        Draw dim random variables uniformly at random from [-epsilon, +epsilon]
        """
        noise = uniform(-epsilon, +epsilon, dim)

        return noise

    @staticmethod
    def compute_fidelity(psi, rho):
        """
        Compute the fidelity according to F(psi, rho) = sqrt(<psi|rho|psi>).
        """
        return sqrt(matmul(matmul(psi.conj().T, rho), psi))

    @staticmethod
    def calculate_expectation_value(counts_0, n_shots):
        """
        Returns the expectation value for a single qubit observable
        given by the total number of shots and the counts of zeros.
        This only works for single qubits, and we assume that the circuit
        has been extended with the appropriate gates, i.e., to measure:
        Z: U - measure
        X: U - H - measure
        Y: U - SGD - H - measure
        """
        return (counts_0 - (n_shots - counts_0)) / n_shots

    @staticmethod
    def reconstruct_density_matrix(exp_x, exp_y, exp_z):
        """
        Reconstructs the density matrix for a single qubit state using the
        Pauli Quantum State Tomography.
        """

        # Reconstruct the density matrix
        rho = 1 / 2 * (Matrices.ID() + exp_x * Matrices.X() + exp_y * Matrices.Y() + exp_z * Matrices.Z())

        return rho

    @staticmethod
    def mirror(arr, negative):
        """
        Mirrors the given array via deleting the middle element and take the
        negative of the values, e.g., [0, 2, 3] -> [-3, -2, 0, 2, 3]
        """
        first = flip(arr)
        second = first[:-1]
        if negative:
            third = multiply(-1, second)
        else:
            third = second
        fourth = append(third, arr)
        return fourth

    @staticmethod
    def write_object_to_file(obj, name):
        dump(obj, open(f"data/{name}.p", "wb"))
        return

    @staticmethod
    def read_object_from_file(name):
        obj = load(open(f"data/{name}.p", "rb"))
        return obj

    @staticmethod
    def read_experiment_from_file(name):
        with open(f"data/{name}.json", mode="r") as f:
            json_string = f.read()
        experiment_results_dict = loads(json_string)
        experiment_results = [ExperimentResult(**d) for d in experiment_results_dict]

        return experiment_results

    @staticmethod
    def write_experiment_to_file(experiment, name):
        json_string = dumps(experiment, sort_keys=True, indent=4)
        with open(f"data/{name}.json", "w", encoding='utf-8') as f:
            f.write(json_string)

        return


class Qiskit:
    @staticmethod
    def get_perfect_simulator():
        return QasmSimulator()

    @staticmethod
    def get_noisy_simulator():
        provider = IBMQ.load_account()
        backend = provider.get_backend(Parameters.IBM_QPU)
        noise_model = NoiseModel.from_backend(backend)
        noisy_simulator = QasmSimulator(noise_model=noise_model)

        return noisy_simulator

    @staticmethod
    def get_real_backend():
        provider = IBMQ.load_account()
        backend = provider.get_backend(Parameters.IBM_QPU)

        return backend

    @staticmethod
    def get_metadata(result):
        """
        Returns the metadata from a result, that is, the name split with "_".
        """

        name = result.header.name

        lip_type, eps, exp_type, instance = name.split("_")

        return lip_type, float(eps), exp_type, int(instance)

    @staticmethod
    def result_to_experiment_result(result):
        """
        Converts a given Qiskit Result to an ExperimentResult.
        """

        counts_dict = result.data.counts

        if len(counts_dict) > 2:
            raise ValueError("Only one qubit is supported.")

        counts = {"0": counts_dict["0x0"], "1": counts_dict["0x1"]}
        lip_type, eps, exp_type, instance = Qiskit.get_metadata(result)
        shots = result.shots
        attributes = {"lip_type": lip_type, "eps": eps, "exp_type": exp_type}

        experiment_result = ExperimentResult(attributes, counts, shots)

        return experiment_result


class Matrices:
    @staticmethod
    def ID():
        return array([[1, 0], [0, 1]], dtype=cdouble)

    @staticmethod
    def X():
        return array([[0, 1], [1, 0]], dtype=cdouble)

    @staticmethod
    def Y():
        return array([[0, -1j], [1j, 0]], dtype=cdouble)

    @staticmethod
    def Z():
        return array([[1, 0], [0, -1]], dtype=cdouble)


@dataclass
class ExperimentResult:
    attributes: dict
    counts: dict
    shots: int
