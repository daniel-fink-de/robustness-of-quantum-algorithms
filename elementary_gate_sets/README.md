# Robustness of the Quantum Fourier Transform for different elementary gate sets

Within this experiment, we illustrate the practical potential 
of the proposed theoretical framework by solving the following 
problem: we study the robustness of different elementary gate 
set implementations of the 3-qubit QFT. To this end, we consider 
the following gate sets:

- Gate set A: 
$\sqrt{X}, X, R_z, CX$ 
(used by [IBM](https://quantum-computing.ibm.com/services/resources))

- Gate set B:
$R_x (\pm \frac{\pi}{2}), R_x (\pm \pi), R_z, CZ$ 
(used by [Rigetti](https://pyquil-docs.rigetti.com/en/v2.7.0/apidocs/gates.html))

- Gate set C:
$U_1, U_2, U_3, CX$ 
(formerly used by [IBM](https://github.com/Qiskit/ibmq-device-information/blob/master/backends/melbourne/V1/README.md))

- Gate set D:
$\sqrt{iSWAP}, FSIM, PhasedXZ, X, Y, Z$
(used by [Google](https://quantumai.google/cirq/google/devices#sycamore))

- Gate set E:
$R_{xy} (\frac{\pi}{2}), R_{xy} (\pi), R_z, U_{zz}$
(used by [Honeywell](https://www.nature.com/articles/s41586-021-03318-4))

This directory consists of 4 Python scripts:

- `bqskit_extensions.py`:
contains extension methods for the BQSKit transpiler to reconstruct 
the Hermitian generators, given a unitary. That is, calculating $H$ from $U=e^{-i H}$.
- `experiment_configuration.py`:
contains useful helper functions and the global parameters for the experiments.
- `transpile.py`:
this is the script that runs the transpilations, simulations and collects the statistics.
- `plot.py`:
after performing the simulations, this script takes the output and plot the figure.

The data associated to this experiment is stored within the `data` directory and contains
- `*.qasm` files for the transpiled circuits
- `*.p` files that store the statistics for one specific transpilation
- `plot.pdf` the figure