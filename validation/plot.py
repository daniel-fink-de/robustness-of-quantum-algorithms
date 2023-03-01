from numpy import array, abs, mean, min, std
import matplotlib as mpl
from matplotlib import pyplot as plt
from experiment_configuration import Functions, Parameters

"""
This script is used to plot the figure.
"""

# Set LaTex
pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],                   # blank entries should cause plots
    "font.sans-serif": [],              # to inherit fonts from the document
    "font.monospace": [],
    "axes.labelsize": 14,               # LaTeX default is 10pt font.
    "font.size": 14,
    "legend.fontsize": 14,               # Make the legend/label fonts
    "xtick.labelsize": 14,               # a little smaller
    "ytick.labelsize": 14,
    "pgf.preamble": "\n".join([ # plots will use this preamble
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{braket}",
        ])
    }
mpl.rcParams.update(pgf_with_latex)


def plot():
    # Read the data
    # axis0 = noise leve, axis1 = instances
    small_simulation = Functions.read_object_from_file("simulation_small")
    large_simulation = Functions.read_object_from_file("simulation_large")
    small_ibm = Functions.read_object_from_file("ibm_small")
    large_ibm = Functions.read_object_from_file("ibm_large")

    # Post-processing (deleting small complex parts)
    small_simulation = abs(small_simulation)
    large_simulation = abs(large_simulation)
    small_ibm = abs(small_ibm)
    large_ibm = abs(large_ibm)

    # Calculate worst points
    small_simulation_worst = min(small_simulation, axis=1)
    large_simulation_worst = min(large_simulation, axis=1)
    small_ibm_worst = min(small_ibm, axis=1)
    large_ibm_worst = min(large_ibm, axis=1)

    # Calculate mean points
    small_simulation_mean = mean(small_simulation, axis=1)
    large_simulation_mean = mean(large_simulation, axis=1)
    small_ibm_mean = mean(small_ibm, axis=1)
    large_ibm_mean = mean(large_ibm, axis=1)

    # Calculate stds
    small_simulation_std = std(small_simulation, axis=1)
    large_simulation_std = std(large_simulation, axis=1)
    small_ibm_std = std(small_ibm, axis=1)
    large_ibm_std = std(large_ibm, axis=1)

    # Calculate theoretical bounds
    small_bounds = [Parameters.BOUND(Parameters.L_SMALL, epsilon) for epsilon in Parameters.SIMULATION_EPSILONS]
    small_bounds = array(small_bounds)
    large_bounds = [Parameters.BOUND(Parameters.L_LARGE, epsilon) for epsilon in Parameters.SIMULATION_EPSILONS]
    large_bounds = array(large_bounds)

    # Figure size
    plt.figure()

    # Plot the grid
    plt.grid()

    # Plot stds for ibm small
    plt.fill_between(Functions.mirror(Parameters.IBM_EPSILONS, True),
                     Functions.mirror(small_ibm_mean - small_ibm_std, False),
                     Functions.mirror(small_ibm_mean + small_ibm_std, False),
                     facecolor="darkorange", alpha=0.25)

    # Plot mean for ibm small
    plt.plot(Functions.mirror(Parameters.IBM_EPSILONS, True),
             Functions.mirror(small_ibm_mean, False),
             color="darkorange", linestyle="solid",
             label="$U_A(\\varepsilon)$ (average)")

    # Plot worst for ibm small
    plt.plot(Functions.mirror(Parameters.IBM_EPSILONS, True),
             Functions.mirror(small_ibm_worst, False),
             color="darkorange", linestyle="dashed",
             label="$U_A(\\varepsilon)$ (worst)")

    # Plot bound for small
    plt.plot(Parameters.SIMULATION_EPSILONS,
             small_bounds,
             color="darkorange", linestyle="dotted",
             label="$1-\\frac{1}{2} L_A^2 \\bar{\\varepsilon}^2$")

    # Plot stds for ibm large
    plt.fill_between(Functions.mirror(Parameters.IBM_EPSILONS, True),
                     Functions.mirror(large_ibm_mean - large_ibm_std, False),
                     Functions.mirror(large_ibm_mean + large_ibm_std, False),
                     facecolor="royalblue", alpha=0.25)

    # Plot mean for ibm large
    plt.plot(Functions.mirror(Parameters.IBM_EPSILONS, True),
             Functions.mirror(large_ibm_mean, False),
             color="royalblue", linestyle="solid",
             label="$U_B(\\varepsilon)$ (average)")

    # Plot worst for ibm large
    plt.plot(Functions.mirror(Parameters.IBM_EPSILONS, True),
             Functions.mirror(large_ibm_worst, False),
             color="royalblue", linestyle="dashed",
             label="$U_B(\\varepsilon)$ (worst)")

    # Plot bound for large
    plt.plot(Parameters.SIMULATION_EPSILONS,
             large_bounds,
             color="royalblue", linestyle="dotted",
             label="$1-\\frac{1}{2} L_B^2 \\bar{\\varepsilon}^2$")

    # Add other details
    plt.xlabel("Noise Level $\\bar{\\varepsilon}$")
    plt.ylabel("Fidelity $\\mathcal{F}(| \\widehat{\\psi} \\rangle, \\rho_{\\varepsilon})$")
    plt.legend(loc='lower center')
    plt.xlim([-1.0, 1.0])
    plt.ylim([0.4, 1.0])

    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig(f'data/plot.pdf')

    return


if __name__ == "__main__":
    plot()
