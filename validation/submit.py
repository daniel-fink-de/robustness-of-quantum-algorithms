from qiskit import QuantumCircuit, transpile
from qiskit.providers.ibmq import IBMQJobManager
from experiment_configuration import Parameters, Qiskit, Functions, Circuits

"""
This skript is used to submit jobs to IBMQ.
"""


def get_circuit(lip, noise, expect, name):
    """
    Generates the circuit based on the lipschitz type (small, large),
    the noise and the expectation value (x, y, z).
    """

    if lip == "small":
        return circuit_small(noise, expect, name)
    if lip == "large":
        return circuit_large(noise, expect, name)
    else:
        return None


def circuit_small(noise, expect, name):
    """
    The circuit with small Lipschitz bound with
    gates for the projective measurements for
    the quantum state tomography.
    """

    qc = Circuits.small(QuantumCircuit(1, name=name), noise)

    if expect == "y":
        qc.sdg(0)
    if expect == "x" or expect == "y":
        qc.h(0)

    qc.measure_all()

    return qc


def circuit_large(noise, expect, name):
    """
    The circuit with large Lipschitz bound with
    gates for the projective measurements for
    the quantum state tomography.
    """

    qc = Circuits.large(QuantumCircuit(1, name=name), noise)

    if expect == "y":
        qc.sdg(0)
    if expect == "x" or expect == "y":
        qc.h(0)

    qc.measure_all()

    return qc


def submit(lip):
    """
    Submit jobs to IBMQ with lip being the circuit to submit, either "small" or "large".
    """

    backend = Qiskit.get_real_backend()

    circuits = []
    for instance in range(1, Parameters.IBM_INSTANCES + 1):
        for epsilon in Parameters.IBM_EPSILONS:
            noise = Functions.draw_noise(epsilon, 1)
            circuits.append(get_circuit(f"{lip}", noise, "x", name=f"{lip}_{epsilon}_x_{instance}"))
            circuits.append(get_circuit(f"{lip}", noise, "y", name=f"{lip}_{epsilon}_y_{instance}"))
            circuits.append(get_circuit(f"{lip}", noise, "z", name=f"{lip}_{epsilon}_z_{instance}"))

    # Need to transpile the circuits first.
    circuits = transpile(circuits, backend=backend)

    # Use Job Manager to break the circuits into multiple jobs.
    job_manager = IBMQJobManager()
    job_set = job_manager.run(circuits, backend=backend, name=f'run_{lip}', shots=Parameters.N_SHOTS)
    job_set_id = job_set.job_set_id()
    print(job_set_id)

    return


if __name__ == "__main__":
    submit("small")
    #submit("large")
