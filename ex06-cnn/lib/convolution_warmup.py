import numpy as np
from typing import Tuple


def pad_array(X: np.ndarray, pad: Tuple[int, int]) -> np.ndarray:
    """Pad the given numpy array with the given padding values.

    Args:
        X: Array of shape (N_batch, n_channels, height, width).
        pad: Tuple of shape (n_height_pad, n_width_pad).

    Returns:
        Array X padded according to parameter pad.
    """

    # START TODO ################
    # Use np.pad to pad the array according to the given padding
    # only pad last two dimensions of X with pad
    pad_width = []
    for i in range(len(X.shape) - 2):
        pad_width.append((0, 0))
    pad_width.append((pad[0], pad[0]))
    pad_width.append((pad[1], pad[1]))
    result = np.pad(X, pad_width=pad_width, mode='constant')
    # END TODO ################
    return result


def dot_product_single_filter(X: np.ndarray, F: np.ndarray, index: Tuple[int, int]) -> float:
    """Calculate the dot product of a small chunk of X with a single filter.

    Args:
        X: Array of shape (n_channels, image_height, image_width).
        F: The filter to apply. Array of shape (n_channels, filter_height, filter_width).
        index: Index of X where the top left corner of the filter should be placed.

    Returns:
        Scalar value which is the sum of product of the filter and section of X.

        For example, consider that your input X is a 5x5 image with only one channel as follows:

        X =
        [[[ 0,  1,  2,  3,  4],
          [ 5,  6,  7,  8,  9],
          [10, 11, 12, 13, 14],
          [15, 16, 17, 18, 19],
          [20, 21, 22, 23, 24]]]

        X.shape = (1, 5, 5)

        and we're using a 3x3 kernel as follows:

        F =
        [[[0, 1, 2],
          [3, 4, 5],
          [6, 7, 8]]]

        F.shape = (1, 3, 3)

        If the given index is (2, 1), then you should compute the sum of the elements of :

        [[[11, 12, 13],              [[[0, 1, 2],
          [16, 17, 18],       *        [3, 4, 5],
          [21, 22, 23]]]               [6, 7, 8]]]

        where * is an elementwise multiplication operation.

        Assume that the kernel will not go outside the boundaries of X. For example, in this case,
        the index (2, 3) is invalid.

        The scalar value that is returned corresponds to a single element in a single activation
        map (see slide 30).
    """

    # START TODO ################
    # starting point of the filter is given by index
    # use broadcasting to multiply the filter with the corresponding section of X
    # and then sum the result

    result = np.sum(X[:, index[0]:index[0] + F.shape[1], index[1]:index[1] + F.shape[2]] * F)
    # END TODO ################

    return result


def dot_product_single_filter_with_batches(X: np.ndarray, F: np.ndarray, index: Tuple[int, int]) -> np.ndarray:
    """Calculate the dot product of a chunk of X with the a single filter, where X has batch_size number of entries.

    Args:
        X: Array of shape (batch_size, n_channels, image_height, image_width).
        F: The filter to apply. Array of shape (n_channels, filter_height, filter_width).
        index: Index of X where the top left corner of the filter should be placed.

    Returns:
        np.ndarray of shape (batch_size, ) which has the sum of product of the filter and section of X
        for each batch.

        This function is practically the same as dot_product_single_filter above but with a new axis in X for the batch.
        Accordingly, it returns a np array of shape (batch_size, ).
    """

    result = np.zeros((len(X)))

    # START TODO ################
    # The filter and section of X no longer have shapes which can be broadcasted together.
    # Use np.expand_dims to fix that.
    # add a new axis to F to make it compatible with X
    result = np.sum(np.expand_dims(F, 0) * X[:, :, index[0]:index[0] + F.shape[1], index[1]:index[1] + F.shape[2]],
                    axis=(1, 2, 3))
    # END TODO ################

    return result


def dot_product_multiple_filters_with_batches(X: np.ndarray, filters: np.ndarray, index: Tuple[int, int]) -> np.ndarray:
    """Calculate the dot product of a chunk of X with a number of filters, where X has batch_size number of entries.

    Args:
        X: Array of shape (batch_size, n_channels, image_height, image_width)
        filters: The filters to apply. Array of shape (n_filters, n_channels, filter_height, filter_width)

    Returns:
        np.ndarray of shape (batch_size, n_filters)

        This function is the same as dot_product_single_filter_with_batches above, except now, we consider more than
        one filters.
    """

    # START TODO ################
    # The filters and section of X no longer have shapes which can be broadcasted together.
    # Use np.expand_dims to fix that.

    # same new axis as before to make F compatible with X
    # but also add a new axis to the result to make it compatible with additional filters
    result = np.sum(np.expand_dims(filters, 1) *
                    np.array(
                        [
                            X[:, :, index[0]:index[0] + filters.shape[-2], index[1]:index[1] + filters.shape[-1]]
                         ] * filters.shape[0]
                    ),
                    axis=(2, 3, 4)).T

    # END TODO ################

    return result
