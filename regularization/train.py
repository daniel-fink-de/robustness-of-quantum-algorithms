from numpy import copy
from numpy.random import seed, uniform
from adam import ADAM
from experiment_configuration import QNN, Parameters, Functions
from dask import config, delayed
from dask.distributed import Client, progress

"""
This script is used to train the models.
"""


# A flag for shutdown the dask cluster
shutdown = False

# A flag for printing the example
print_example = True

# Set the scheduler to use parallelization
config.set(scheduler='threads')


def train_one_instance(lamb, eps, seed_val):
    """
    Perform the training of one instance.
    """

    best_cost = 1000.0
    best_weights = None

    # draw weights randomly
    seed(seed_val)
    weights = uniform(Parameters.INITIAL_WEIGHT_INTERVAL[0],
                      Parameters.INITIAL_WEIGHT_INTERVAL[1],
                      Parameters.TRAIN_SHAPE)

    optimizer = ADAM(Parameters.N_PARAMETERS, alpha=Parameters.INITIAL_LR)

    for i_iteration in range(Parameters.ITERATIONS):
        # calculate the real cost (since we compare that)
        cost = QNN.cost(Parameters.DATA_X, weights, Parameters.DATA_Y, eps, lamb)

        if best_cost > cost:
            best_cost = cost
            best_weights = copy(weights)

        # calculate gradient
        grad = QNN.grad_f(Parameters.DATA_X, weights, Parameters.DATA_Y, eps, lamb)

        # perform optimization
        weights = optimizer.step(weights, grad)

    name = f"weights_lamb_{lamb}_eps_{eps}_seed_{seed_val}"
    Functions.write_object_to_file(best_weights, name)

    return


def train():
    client = Client()
    if shutdown:
        client.shutdown()
        exit()

    for i_lambda, lamb in enumerate(Parameters.LAMBDAS):
        print(f"Running with lambda {lamb}")
        for i_epsilon, eps in enumerate(Parameters.EPSILONS):
            print(f"Running with noise level {eps}")

            delayed_function = delayed(train_one_instance)
            delayed_results = [delayed_function(lamb, eps, seed_val) for seed_val in Parameters.SEEDS]

            results = client.compute(delayed_results)
            progress(results)

    return


if __name__ == "__main__":
    train()
