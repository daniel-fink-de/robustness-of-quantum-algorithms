from numpy import zeros, cdouble
from numpy.random import seed
from qiskit import QuantumCircuit, transpile
from dask import config, delayed
from dask.distributed import Client, progress
from experiment_configuration import Parameters, Circuits, Functions, Qiskit

"""
This skript runs simulations of the circuits on IBM noisy simulators, calculates the fidelities
and stores them in files.
"""

# A flag for shutdown the dask cluster
shutdown = False

# A flag for printing the example
print_example = True

# Set the scheduler to use parallelization
config.set(scheduler='threads')


def simulate():
    """
    Simulate the circuits.
    """

    client = Client()

    if shutdown:
        client.shutdown()
        exit()

    seed(Parameters.NUMPY_SEED)

    psi = Circuits.perfect_state()
    simulator = Qiskit.get_noisy_simulator()

    if print_example:
        print_an_example(Qiskit.get_perfect_simulator())

    jobs = []
    for epsilon in Parameters.SIMULATION_EPSILONS:
        job = delayed(simulate_for_one_noise_level)(Parameters.SIMULATION_INSTANCES, psi, epsilon, simulator)
        jobs.append(job)

    aggregation = delayed(aggregate)(len(Parameters.SIMULATION_EPSILONS), Parameters.SIMULATION_INSTANCES, jobs)

    aggregation = aggregation.persist()  # start computation in the background
    progress(aggregation)  # watch progress
    fidelities_small, fidelities_large = aggregation.compute()

    Functions.write_object_to_file(fidelities_small, "simulation_small")
    Functions.write_object_to_file(fidelities_large, "simulation_large")

    return


def aggregate(n_epsilons, n_instances, list_of_fidelity_tuples):
    """
    Aggregates the dask jobs.
    """
    fidelities_small = zeros((n_epsilons, n_instances), dtype=cdouble)
    fidelities_large = zeros((n_epsilons, n_instances), dtype=cdouble)

    for i in range(n_epsilons):
        fidelities_small[i, :] = list_of_fidelity_tuples[i][0]
        fidelities_large[i, :] = list_of_fidelity_tuples[i][1]

    return fidelities_small, fidelities_large


def print_an_example(simulator):
    noise = Functions.draw_noise(max(Parameters.SIMULATION_EPSILONS), 1)
    qc_small = Circuits.small(QuantumCircuit(1), noise)
    qc_small.save_density_matrix()
    qc_large = Circuits.large(QuantumCircuit(1), noise)
    qc_large.save_density_matrix()

    qc_small_transpiled = transpile(qc_small, simulator, optimization_level=Parameters.OPTIMIZATION_LEVEL)
    qc_large_transpiled = transpile(qc_large, simulator, optimization_level=Parameters.OPTIMIZATION_LEVEL)

    print(f"Noise = {noise}")
    print("Small circuit:")
    print(qc_small_transpiled.draw())
    print("Large circuit:")
    print(qc_large_transpiled.draw())

    return


def simulate_for_one_noise_level(n_instances, psi, epsilon, simulator):
    """
    Run several instances for one epsilon.
    NOTE: The randomly chosen noise in (-epsilon,+epsilon)
    is the same for all shots of the circuit,
    and it will be in each instance the same for
    the small and large circuit.
    """
    fidelities_small = zeros(n_instances, dtype=cdouble)
    fidelities_large = zeros(n_instances, dtype=cdouble)

    for i in range(n_instances):
        # Sample a noise-term
        noise = Functions.draw_noise(epsilon, 1)

        # Get the density matrix for small
        qc = QuantumCircuit(1)
        qc = Circuits.small(qc, noise)
        qc.save_density_matrix()
        qc = transpile(qc, simulator, optimization_level=Parameters.OPTIMIZATION_LEVEL)
        job = simulator.run(qc, shots=Parameters.N_SHOTS)
        rho = job.result().data()["density_matrix"].data
        # Calculate fidelity
        fidelity = Functions.compute_fidelity(psi, rho)
        fidelities_small[i] = fidelity

        # Get the density matrix for large
        qc = QuantumCircuit(1)
        qc = Circuits.large(qc, noise)
        qc.save_density_matrix()
        qc = transpile(qc, simulator, optimization_level=Parameters.OPTIMIZATION_LEVEL)
        job = simulator.run(qc, shots=Parameters.N_SHOTS)
        rho = job.result().data()["density_matrix"].data
        # Calculate fidelity
        fidelity = Functions.compute_fidelity(psi, rho)
        fidelities_large[i] = fidelity

    return fidelities_small, fidelities_large


if __name__ == "__main__":
    simulate()
