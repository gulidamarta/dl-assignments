"""Plotting functions."""

import matplotlib.pyplot as plt
import numpy as np

from lib.lr_schedulers import PiecewiseConstantLR, CosineAnnealingLR
from lib.optimizers import Adam, SGD
from lib.network_base import Parameter
from lib.utilities import load_result


def plot_learning_curves() -> None:
    """Plot the performance of SGD, SGD with momentum, and Adam optimizers.

    Note:
        This function requires the saved results of compare_optimizers() above, so make
        sure you run compare_optimizers() first.
    """
    optim_results = load_result('optimizers_comparison')
    # START TODO ################
    # train result are tuple(train_costs, train_accuracies, eval_costs,
    # eval_accuracies). You can access the iterable via
    # optim_results.items()
    x = np.linspace(0, 10, 10)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    for item in optim_results.items():
        ax1.plot(x, item[1][0], label=item[0])
        ax2.plot(x, item[1][1], label=item[0])
    ax1.set_title("Model comparison on training set - Cost and Accuracy")
    ax1.set(ylabel="Loss")
    ax2.set(xlabel='Epochs', ylabel="Accuracy")
    ax1.legend()
    ax2.legend()
    plt.show()
    # END TODO ###################


def plot_lr_schedules() -> None:
    """Plot the learning rate schedules of piecewise and cosine schedulers.

    """
    num_epochs = 80
    base_lr = 0.1

    piecewise_scheduler = PiecewiseConstantLR(Adam([], lr=base_lr), [10, 20, 40, 50], [0.1, 0.05, 0.01, 0.001])
    cosine_scheduler = CosineAnnealingLR(Adam([], lr=base_lr), num_epochs)

    # START TODO ################
    piecewise_scheduler_lrs = []
    cosine_scheduler_lrs = []
    # get the learning rates
    for i in range(num_epochs):
        piecewise_scheduler.step()
        cosine_scheduler.step()
        piecewise_scheduler_lrs.append(piecewise_scheduler.optimizer.lr)
        cosine_scheduler_lrs.append(cosine_scheduler.optimizer.lr)
    # plot piecewise lr and cosine lr
    x = np.linspace(0, 80, 80)
    plt.plot(x, piecewise_scheduler_lrs, label=" piecewise_scheduler")
    plt.plot(x, cosine_scheduler_lrs, label=" cosine_scheduler")
    plt.xlabel("Epochs")
    plt.ylabel("Learning rates")
    plt.show()
    # END TODO ################
