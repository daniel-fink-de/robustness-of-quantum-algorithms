# Validation on a quantum computer

Within this experiment, we validate our theoretical findings
in an implementation on the [ibm_nairobi](https://quantum-computing.ibm.com)
quantum computer.
To this end, we consider the two following quantum circuits

- $\hat{U}_A \ket{0} = \sqrt{X} R_z(\frac{\pi}{4}) \sqrt{X} \ket{0}$
- $\hat{U}_B \ket{0} = \sqrt{X} R_z(\frac{5\pi}{4}) \sqrt{X} X \ket{0}$

which are affected via coherent control errors according to

- $U_A(\varepsilon_A) \ket{0} = \sqrt{X} R_z(\frac{\pi}{4}(1+\varepsilon_A)) \sqrt{X} \ket{0}$
- $U_B(\varepsilon_B) \ket{0} = \sqrt{X} R_z(\frac{5\pi}{4}(1+\varepsilon_B)) \sqrt{X} X \ket{0}$.

This directory consists of 6 Python scripts:

- `experiment_configuration.py`:
contains useful helper functions and the global parameters for the experiments.
- `experiment_configuration.py`:
a script for submitting jobs to IBMQ.
- `retrieve.py`:
a script for retrieving jobs from IBMQ.
- `calculate.py`:
calculates the fidelities based on the raw experiment dat from IBMQ.
- `simulate.py`:
this script can be used to perform simulations with noisy backends rather than using a real qpu.
- `plot.py`:
after performing the simulations or real experiments, this script takes the output and plot the figure.

The data associated to this experiment is stored within the `data` directory and contains
- `ibm.json` file that consists of all the raw experiment results performed with ibm_nairobi.
- `*.p` files that store the statistics (fidelities) after the preprocessing of the raw experiment data.
- `plot.pdf` the figure