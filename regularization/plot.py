from numpy import array, abs, mean, sum, std
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
    # lambda, epsilon
    bounds = []

    lambdas = [0.0, 0.01, 0.05, 0.1, 0.5]
    labels = [f"$\\lambda={str(l)}$" for l in lambdas]
    colors = ['darkorange', 'royalblue', 'gray', 'tomato', 'yellowgreen']
    epsilon = 0.05

    # read the data
    for i_lamb, lamb in enumerate(lambdas):
        weights_all = []
        for i_seed, seed_val in enumerate(Parameters.SEEDS):
            name = f"weights_lamb_{lamb}_eps_{epsilon}_seed_{seed_val}"
            weights = Functions.read_object_from_file(name)
            # calculate lip bound, i.e., (1/2) |theta1| + (1/2) |theta2| + ...
            weights = 0.5 * sum(abs(weights))
            weights_all.append(weights)
        # we store the array over all instances
        bounds.append(array(weights_all))

    means = [mean(b) for b in bounds]
    stds = [std(b) for b in bounds]

    # Create the figure and axis
    fig, ax = plt.subplots()
    ax.bar(labels, means, yerr=stds,
           align='center',
           alpha=0.75,
           ecolor='saddlebrown',
           capsize=10,
           color=colors)

    # Add labels and title
    plt.gca().yaxis.grid(True)
    plt.gca().set_axisbelow(True)

    plt.ylabel("Lipschitz Bound")
    plt.ylim([0, 8])
    y_tick_labels = ["0.0", "1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0"]
    y_tick_values = [float(y) for y in y_tick_labels]

    plt.yticks(y_tick_values, y_tick_labels)

    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig(f'data/plot.pdf')

    return


if __name__ == "__main__":
    plot()