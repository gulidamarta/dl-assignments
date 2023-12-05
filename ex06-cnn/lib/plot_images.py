import matplotlib.pyplot as plt
import numpy as np


def plot_data(data: np.ndarray, rows: int = 5, cols: int = 4, plot_border: bool = True, title: str = "") -> None:
    """Plot the given image data.

    Args:
        data: image data shaped (n_samples, channels, width, height).
        rows: number of rows in the plot .
        cols: number of columns in the plot.
        plot_border: add a border to the plot of each individual digit.
                     If True, also disable the ticks on the axes of each image.
        title: add a title to the plot.

    Returns:
        None

    Note:

    """
    # START TODO ################
    # useful functions: plt.subplots, plt.suptitle, plt.imshow
    fig, ax = plt.subplots(rows, cols)
    fig.suptitle(title)
    for i in range(rows):
        for j in range(cols):
            ax[i, j].imshow(data[i * cols + j][0])
            if plot_border:
                ax[i, j].set_axis_off()
    plt.show()

    # END TODO ################
