from experiment_configuration import Circuits, Functions
from collections import defaultdict
from queue import Queue
from numpy import zeros, cdouble

"""
This skript takes the raw experiment data and calculates fidelities based on that.
"""


def calculate():
    """
    Calculate the fidelities and stores them in files.
    """

    psi = Circuits.perfect_state()
    experiment_results = Functions.read_experiment_from_file("ibm")

    lip_types = defaultdict(list)
    eps_values = defaultdict(list)
    exp_types = defaultdict(list)

    # Sort all results based on their attributes
    for result in experiment_results:
        lip_type = result.attributes["lip_type"]
        eps_value = float(result.attributes["eps"])
        exp_type = result.attributes["exp_type"]

        lip_types[lip_type].append(result)
        eps_values[eps_value].append(result)
        exp_types[exp_type].append(result)

    # Get all values of epsilon
    eps = [i for i in eps_values if eps_values[i] != eps_values.default_factory()]
    eps.sort()

    # process small
    small_fidelities = None
    small_experiments = lip_types["small"]
    for i, epsilon in enumerate(eps):
        small_experiments_with_epsilon = [e for e in small_experiments if float(e.attributes["eps"]) == epsilon]
        x_queue = Queue()
        y_queue = Queue()
        z_queue = Queue()
        for experiment in small_experiments_with_epsilon:
            if experiment.attributes["exp_type"] == "x":
                x_queue.put(experiment)
            elif experiment.attributes["exp_type"] == "y":
                y_queue.put(experiment)
            elif experiment.attributes["exp_type"] == "z":
                z_queue.put(experiment)
            else:
                raise ValueError()

        if not (x_queue.qsize() == y_queue.qsize() == z_queue.qsize()):
            raise ValueError()

        n_instances = x_queue.qsize()
        if small_fidelities is None:
            small_fidelities = zeros((len(eps), n_instances), dtype=cdouble)

        for j in range(x_queue.qsize()):
            # Reconstruct one density matrix
            experiment = x_queue.get()
            counts_0 = experiment.counts["0"]
            n_shots = experiment.shots
            exp_x = Functions.calculate_expectation_value(counts_0, n_shots)
            experiment = y_queue.get()
            counts_0 = experiment.counts["0"]
            n_shots = experiment.shots
            exp_y = Functions.calculate_expectation_value(counts_0, n_shots)
            experiment = z_queue.get()
            counts_0 = experiment.counts["0"]
            n_shots = experiment.shots
            exp_z = Functions.calculate_expectation_value(counts_0, n_shots)
            rho = Functions.reconstruct_density_matrix(exp_x, exp_y, exp_z)

            # calculate fidelity
            fidelity = Functions.compute_fidelity(psi, rho)
            small_fidelities[i, j] = fidelity

    # process large
    large_fidelities = None
    large_experiments = lip_types["large"]
    for i, epsilon in enumerate(eps):
        large_experiments_with_epsilon = [e for e in large_experiments if float(e.attributes["eps"]) == epsilon]
        x_queue = Queue()
        y_queue = Queue()
        z_queue = Queue()
        for experiment in large_experiments_with_epsilon:
            if experiment.attributes["exp_type"] == "x":
                x_queue.put(experiment)
            elif experiment.attributes["exp_type"] == "y":
                y_queue.put(experiment)
            elif experiment.attributes["exp_type"] == "z":
                z_queue.put(experiment)
            else:
                raise ValueError()

        if not (x_queue.qsize() == y_queue.qsize() == z_queue.qsize()):
            raise ValueError()

        n_instances = x_queue.qsize()
        if large_fidelities is None:
            large_fidelities = zeros((len(eps), n_instances), dtype=cdouble)

        for j in range(x_queue.qsize()):
            # Reconstruct one density matrix
            experiment = x_queue.get()
            counts_0 = experiment.counts["0"]
            n_shots = experiment.shots
            exp_x = Functions.calculate_expectation_value(counts_0, n_shots)
            experiment = y_queue.get()
            counts_0 = experiment.counts["0"]
            n_shots = experiment.shots
            exp_y = Functions.calculate_expectation_value(counts_0, n_shots)
            experiment = z_queue.get()
            counts_0 = experiment.counts["0"]
            n_shots = experiment.shots
            exp_z = Functions.calculate_expectation_value(counts_0, n_shots)
            rho = Functions.reconstruct_density_matrix(exp_x, exp_y, exp_z)

            # calculate fidelity
            fidelity = Functions.compute_fidelity(psi, rho)
            large_fidelities[i, j] = fidelity

    Functions.write_object_to_file(small_fidelities, "ibm_small")
    Functions.write_object_to_file(large_fidelities, "ibm_large")

    return


if __name__ == "__main__":
    calculate()
