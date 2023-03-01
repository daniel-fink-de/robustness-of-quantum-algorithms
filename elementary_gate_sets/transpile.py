from numpy import zeros, dot, cdouble
from bqskit import MachineModel, compile
from bqskit.ext import qiskit_to_bqskit
from experiment_configuration import Gates, Circuits, Parameters, Functions, Experiment

"""
This skript transpiles the QFT from Qiskit into its representation using different gate sets.
"""


def print_experiment_statistics(experiment):
    """
    Print some statistics about the experiment.
    """

    print(f"{experiment.gate_set_name}: Lipschitz bound {experiment.bound}")
    print(f"{experiment.gate_set_name}: Number of gates {experiment.n_gates}")
    print(f"{experiment.gate_set_name}: Depth {experiment.depth}")

    return


def get_machine_model(gate_set_name):
    """
    Get the machine model based on the name of the gate set.
    """

    if gate_set_name == "a":
        return MachineModel(Parameters.N_QUBITS, gate_set=Gates.A())
    elif gate_set_name == "b":
        return MachineModel(Parameters.N_QUBITS, gate_set=Gates.B())
    elif gate_set_name == "c":
        return MachineModel(Parameters.N_QUBITS, gate_set=Gates.C())
    elif gate_set_name == "d":
        return MachineModel(Parameters.N_QUBITS, gate_set=Gates.D())
    elif gate_set_name == "e":
        return MachineModel(Parameters.N_QUBITS, gate_set=Gates.E())
    else:
        return None


def compile_circuit(circuit, gate_set_name):
    """
    Compile to a bqskit circuit with give string for the gate set.
    """

    machine_model = get_machine_model(gate_set_name)

    circuit = compile(circuit,
                      model=machine_model,
                      optimization_level=Parameters.OPTIMIZATION_LEVEL,
                      max_synthesis_size=Parameters.MAX_SYNTHESIS_SIZE,
                      synthesis_epsilon=Parameters.SYNTHESIS_EPSILON,
                      error_threshold=Parameters.SYNTHESIS_EPSILON)

    return circuit


def run_instances_with_compiled_circuit(compiled_circuit, perfect_unitary, gate_set_name):
    fidelities = zeros(Parameters.N_INSTANCES, dtype=cdouble)
    bound = 0.0

    for i_instance in range(Parameters.N_INSTANCES):

        if i_instance % 500 == 0:
            print(f"{gate_set_name}: Instance {i_instance}/{Parameters.N_INSTANCES}")

        # get the noise-free result
        psi = Functions.draw_random_state(Parameters.N_QUBITS)
        phi = dot(perfect_unitary, psi)

        # get the noisy unitary
        noisy_unitary, bound = Circuits.to_noisy_unitary(compiled_circuit, Parameters.EPSILON)

        # get noisy result
        phi_noisy = dot(noisy_unitary, psi)
        fidelity = Functions.compute_fidelity(phi, phi_noisy)

        # store fidelity
        fidelities[i_instance] = fidelity

    experiment = Experiment(
        gate_set_name=gate_set_name,
        fidelities=fidelities,
        n_gates=compiled_circuit.num_operations,
        bound=bound,
        depth=compiled_circuit.depth)

    Functions.write_object_to_file(experiment, gate_set_name)

    return


def transpile():
    # Get the circuit
    qc = Circuits.perfect()

    # Draw the circuit
    print(qc.draw())

    # Convert the circuit to bqskit as base
    circuit = qiskit_to_bqskit(qc)
    perfect_unitary = circuit.get_unitary()

    # Write ideal circuit to file
    Functions.save_circuit_to_file(circuit, "qft")

    # Compile the circuits
    compiled_circuits = []
    for gate_set_name in Parameters.GATE_SET_NAMES:
        compiled_circuit = compile_circuit(circuit, gate_set_name)
        compiled_circuits.append(compiled_circuit)

    for i_gate_set_name, gate_set_name in enumerate(Parameters.GATE_SET_NAMES):
        print(f"{gate_set_name}:")
        print(f"Gates = {compiled_circuits[i_gate_set_name].num_operations}")
        print(f"Depths = {compiled_circuits[i_gate_set_name].depth}")

    for i_gate_set, gate_set_name in enumerate(Parameters.GATE_SET_NAMES):
        compiled_circuit = compiled_circuits[i_gate_set]
        # Write compiled circuit to file
        Functions.save_circuit_to_file(compiled_circuit, f"qft_{gate_set_name}")
        run_instances_with_compiled_circuit(compiled_circuit, perfect_unitary, gate_set_name)

    return


if __name__ == "__main__":
    transpile()
