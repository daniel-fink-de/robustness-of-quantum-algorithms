from numpy import arange, zeros, mean, min, std, abs
import matplotlib as mpl
from matplotlib import pyplot as plt
from experiment_configuration import Functions

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
    experiment_a = Functions.read_object_from_file("a")
    experiment_b = Functions.read_object_from_file("b")
    experiment_c = Functions.read_object_from_file("c")
    experiment_d = Functions.read_object_from_file("d")
    experiment_e = Functions.read_object_from_file("e")

    # all experiments
    experiments = [
        experiment_a,
        experiment_b,
        experiment_c,
        experiment_d,
        experiment_e,
    ]
    labels = [
        f"A ({experiments[0].n_gates}, {experiments[0].depth})",
        f"B ({experiments[1].n_gates}, {experiments[1].depth})",
        f"C ({experiments[2].n_gates}, {experiments[2].depth})",
        f"D ({experiments[3].n_gates}, {experiments[3].depth})",
        f"E ({experiments[4].n_gates}, {experiments[4].depth})",
    ]

    # set width of bars
    barWidth = 0.25

    # set heights of bars
    fidelities_mean = [mean(abs(experiment.fidelities)) for experiment in experiments]
    fidelities_std = [std(abs(experiment.fidelities)) for experiment in experiments]
    fidelities_worst = [min(abs(experiment.fidelities)) for experiment in experiments]
    bounds = [experiment.bound for experiment in experiments]

    # Set position of bar on X axis
    r_mean = arange(len(fidelities_mean))
    r_worst = [x + barWidth for x in r_mean]
    r_bound = [x + barWidth for x in r_worst]

    # Make the plot
    plt.bar(r_mean, fidelities_mean, color="darkorange", alpha=0.75, width=barWidth, edgecolor='white', label='Average')
    plt.bar(r_worst, fidelities_worst, color="royalblue", alpha=0.75, width=barWidth, edgecolor='white', label='Worst')

    # Make ghost plots for second axis
    plt.bar(r_bound, zeros(len(fidelities_mean)), color='gray', alpha=0.75, width=barWidth, edgecolor='white', label='$L_{QFT}$')

    # Plot std for fidelity mean
    plt.errorbar(r_mean, fidelities_mean, yerr=fidelities_std, color="saddlebrown", fmt='o', markersize=0, capsize=4, elinewidth=1.0)

    plt.gca().yaxis.grid(True)
    plt.gca().set_axisbelow(True)

    # Add xticks on the middle of the group bars
    plt.xlabel('group', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(fidelities_mean))], labels)

    # Add other details
    plt.xlabel(f"")
    plt.ylabel("Fidelity $|\\langle \\psi(\\varepsilon) | \\widehat{\\psi} \\rangle|$")
    plt.ylim([0.68, 1.0])

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), fancybox=False, shadow=False, ncol=5)

    y_tick_labels = ["0.68", "0.72", "0.76", "0.80", "0.84", "0.88", "0.92", "0.96", "1.00"]
    y_tick_values = [float(y) for y in y_tick_labels]

    plt.yticks(y_tick_values, y_tick_labels)

    # Create second y axis
    ax2 = plt.gca().twinx()

    # Plot additional bar on second y axis
    plt.bar(r_bound, bounds, color="gray", alpha=0.75, edgecolor='white', width=barWidth)

    # Add labels for second y axis
    ax2.set_ylabel("Lipschitz Bound")
    ax2.set_ylim([0, 160])

    y_gate_tick_labels = ["0", "20", "40", "60", "80", "100", "120", "140", "160"]
    y_gate_tick_values = [float(y) for y in y_gate_tick_labels]
    ax2.set_yticks(y_gate_tick_values, y_gate_tick_labels)

    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig(f'data/plot.pdf', bbox_inches='tight')

    return


if __name__ == "__main__":
    plot()
