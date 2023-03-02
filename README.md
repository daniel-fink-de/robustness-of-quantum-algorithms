# Robustness of quantum algorithms against coherent control errors
This repository contains the source code for the paper 
"Robustness of quantum algorithms against coherent control errors" 
by Julian Berberich, Daniel Fink and Christian Holm [[arXiv:2303.00618](https://arxiv.org/abs/2303.00618)].

The source code uses Python 3.10. 
All dependencies can be installed using the `requirements.txt` file.

In total, the repository consists of three directories, 
each corresponding to one of the three experiments from the sections 5, 6 and 7 of the paper.

### Robustness of the Quantum Fourier Transform for different elementary gate sets
- Directory: `elementary_gate_set`

Within this experiment, we illustrate the practical potential of the proposed theoretical framework 
by solving the following problem: 
we study the robustness of different elementary gate set implementations of the 3-qubit QFT. 

### Validation on a quantum computer
- Directory: `validation`

Here, we validate our theoretical findings in an implementation on the 
[ibm_nairobi](https://quantum-computing.ibm.com) quantum computer.

### Variational quantum algorithms: robustness via regularization
- Directory: `regularization`

In this experiment, we verify that our theoretical framework
can be used to successfully improve robustness of quantum
circuits in the context of VQAs via regularization.
