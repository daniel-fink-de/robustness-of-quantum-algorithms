# Variational quantum algorithms: robustness via regularization

Within this experiment, we verify that our theoretical framework
can be used to successfully improve robustness of quantum
circuits in the context of VQAs via regularization.
To do so, we employ a simple regression task. 
That is, we are interested in learning a regression
function $f_{\theta}(x)$ based on a given data set 
$$\mathcal{D} = \\{ (x,y) : x \in \mathcal{X}, y \in \mathcal{Y} \\}.$$
To be more precise, we choose $\mathcal{X}$ as $20$ points equidistant from $[0, 2\pi]$
and a target set $$\mathcal{Y} = \\{ sin(x) : x \in \mathcal{X} \\}.$$

As leanring model, we employ the single qubit trainable quantum circuit
$U(x, \theta) = R_x(x) R_z(\theta_1) R_y(\theta_2) R_z(\theta_3)$,
which is based on [Effect of data encoding](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.103.032430).
The parameters $\theta = (\theta_1, \theta_2, \theta_3)$ are updated using [ADAM](https://arxiv.org/abs/1412.6980) optimizer.

This directory consists of 4 Python scripts:

- `experiment_configuration.py`:
contains useful helper functions and the global parameters for the experiments.
- `adam.py`:
containing the implementation of the ADAM optimizer.
- `train.py`:
a script used to train the regression models.
- `plot.py`:
after performing the trainings, this script takes the output and plot the figure.

The data associated to this experiment is stored within the `data` directory and contains
- `*.p` files that store the trained weights for a learned model instance.
- `plot.pdf` the figure