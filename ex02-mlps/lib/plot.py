"""ReLU plot."""

import matplotlib.pyplot as plt
import numpy as np

from lib.activations import ReLU


def plot_relu() -> None:
    """Plot the ReLU function in the range (-4, 4).

    Returns:
        None
    """
    # START TODO #################
    x = np.linspace(-4, +4, 200)

    relu = ReLU()
    y = relu.forward(x)

    # plot input and relu output
    plt.plot(x, y)
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.grid(True)
    plt.show()
    # END TODO###################
