from dataclasses import dataclass
from pickle import dump, load
from numpy import pi, copy, zeros, linspace, sin, double, sum, square
from numpy.random import uniform
from qiskit import QuantumCircuit
from qiskit_aer import QasmSimulator

"""
This module contains global experiment configurations.
"""


@dataclass
class Parameters:
    """
    Stores the parameters for the entire experiment.
    """

    N_QUBITS = 1
    N_LAYERS = 1
    EPSILONS = [0.05]
    INITIAL_WEIGHT_INTERVAL = [-2*pi, +2*pi]
    SEEDS = [12, 34331, 166, 3433, 31, 90, 345, 5315]
    N_SHOTS = 20_000
    IBM_KEY = ""  # put the IBM key here
    SIMULATOR = QasmSimulator()
    TRAIN_SHAPE = (N_LAYERS, N_QUBITS, 3)
    N_PARAMETERS = N_LAYERS * N_QUBITS * 3
    ITERATIONS = 51
    ITERATIONS_PRINTER = 10
    DATA_X = linspace(0, 2*pi, 20)
    DATA_Y = sin(DATA_X)
    LAMBDAS = [0.0, 0.01, 0.05, 0.1, 0.5]
    INITIAL_LR = 0.05


class QNN:
    @staticmethod
    def f(x, w, noise=None):
        """
        This is a map f_w(x) = y, where w are the weights and x is the vector containing all input data.
        The weights have the shape (n_layers, n_qubits, 3).
        If noise is not None, it will be treated as the noise level.
        """

        exp_z = zeros(len(x), dtype=double)

        for i_data in range(len(x)):
            val = QNN.f_single_data_point(x[i_data], w, noise)
            exp_z[i_data] = val

        return exp_z

    @staticmethod
    def f_single_data_point(x, w, noise=None):
        """
        This is a map f_w(x) = y, where w are the weights and x is the scalar input data.
        The weights have the shape (n_layers, n_qubits, 3).
        If noise is not None, it will be treated as the noise level.
        """

        # Create the circuit
        qc = QuantumCircuit(Parameters.N_QUBITS, 1)

        # Insert noise now
        if noise is not None:
            errors = Functions.draw_noise(noise)
            w = copy(w)
            w = (1.0 + errors) * w

        for i_layer in range(Parameters.N_LAYERS):
            # trainable block
            for i_qubit in range(Parameters.N_QUBITS):
                qc.rz(w[i_layer, i_qubit, 0], i_qubit)
                qc.ry(w[i_layer, i_qubit, 1], i_qubit)
                qc.rz(w[i_layer, i_qubit, 2], i_qubit)

            # entangling
            if Parameters.N_QUBITS > 1:
                qc.cx(0, 1)
            elif Parameters.N_QUBITS > 2:
                for i_qubit in range(Parameters.N_QUBITS - 1):
                    qc.cx(i_qubit, i_qubit + 1)
                qc.cx(Parameters.N_QUBITS - 1, 0)

            # data encoding
            for i_qubit in range(Parameters.N_QUBITS):
                qc.rx(x, i_qubit)

        qc.measure(0, 0)

        job = Parameters.SIMULATOR.run(qc, shots=Parameters.N_SHOTS)
        if "0" in job.result().get_counts().keys():
            counts0 = job.result().get_counts()["0"]
        else:
            counts0 = 0.0
        exp_z = Functions.calculate_z_expectation_value(counts0)

        return exp_z

    @staticmethod
    def grad_f(x, w, y, noise=None, reg=None):
        """
        Calculate the gradient of the MSE, i.e.,

        grad 1/n sum_i=1^n [y_i - f_w(x_i)]^2 + lambda * sum_j^m w_j^2

        with respect to w. Without the regularization, this is

        grad = - 2 * 1/n sum_i=1^n [y_i - f_w(x_i)] * grad f_w(x_i),

        f_w(x_i) = <psi(x,w)|Z|psi(x,w)>

        and with regularization, this is

        grad = - 2 * 1/n sum_i=1^n [y_i - f_w(x_i)] * grad f_w(x_i) + 2 * lambda * w.
        """

        # calculate the gradient per parameter and per data point
        gradients = zeros((len(x), Parameters.N_LAYERS, Parameters.N_QUBITS, 3), dtype=double)
        for i_data in range(len(x)):
            for i_layer in range(Parameters.N_LAYERS):
                for i_qubit in range(Parameters.N_QUBITS):
                    for i_rotation in range(3):
                        w_p = copy(w)
                        w_p[i_layer, i_qubit, i_rotation] = w_p[i_layer, i_qubit, i_rotation] + pi / 2
                        w_m = copy(w)
                        w_m[i_layer, i_qubit, i_rotation] = w_m[i_layer, i_qubit, i_rotation] - pi / 2

                        grad_for_single_weight = 0.5 * \
                                                 (QNN.f_single_data_point(x[i_data], w_p, noise=noise)
                                                  - QNN.f_single_data_point(x[i_data], w_m, noise=noise))

                        gradients[i_data, i_layer, i_qubit, i_rotation] = grad_for_single_weight

        # aggregate the gradient over all data points
        gradient = zeros((Parameters.N_LAYERS, Parameters.N_QUBITS, 3), dtype=double)
        for i_data in range(len(x)):
            for i_layer in range(Parameters.N_LAYERS):
                for i_qubit in range(Parameters.N_QUBITS):
                    for i_rotation in range(3):
                        evaluation = QNN.f_single_data_point(x[i_data], w, noise=noise)

                        gradient[i_layer, i_qubit, i_rotation] += (y[i_data] - evaluation) \
                                                                  * gradients[i_data, i_layer, i_qubit, i_rotation]

        gradient = gradient * (-2.0 / len(x))

        if reg is not None:
            for i_layer in range(Parameters.N_LAYERS):
                for i_qubit in range(Parameters.N_QUBITS):
                    for i_rotation in range(3):
                        regularization_term = 2.0 * reg * w[i_layer, i_qubit, i_rotation]
                        gradient[i_layer, i_qubit, i_rotation] += regularization_term

        return gradient

    @staticmethod
    def cost(x, w, y, noise=None, reg=None):
        """
        Evaluate the cost function (MSE), that is

        cost = 1/n sum_i=1^n [y_i - f_w(x_i)]^2 + lambda * sum_j^m |w_j|^2

        with regularization lambda or no regularization (lambda = None).
        """

        # get the evaluations f_w(x_i)
        evaluations = QNN.f(x, w, noise)

        cost = 0.0

        for i_data in range(len(x)):
            cost += (y[i_data] - evaluations[i_data]) ** 2

        cost = cost * (1.0 / len(x))

        if reg is not None:
            cost = cost + reg * sum(square(w))

        return cost


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
    def calculate_z_expectation_value(counts0):
        """
        Calculate the <Z> expectation value given the counts of 0 for a single qubit measurement.
        """

        counts1 = Parameters.N_SHOTS - counts0

        exp_z = (counts0 - counts1) / (counts0 + counts1)

        return exp_z

    @staticmethod
    def draw_noise(noise_level):
        """
        Draw noise of the shape of the weights uniformly at random.
        """

        noise = uniform(-noise_level, noise_level, Parameters.TRAIN_SHAPE)

        return noise
