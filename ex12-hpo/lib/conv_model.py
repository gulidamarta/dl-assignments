from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn


def get_conv_model(num_filters_per_layer: List[int]) -> nn.Module:
    """
    Builds a deep convolutional model with varying number of convolutional
    layers (and # filters per layer) for MNIST input using pytorch.

    Args:
        num_filters_per_layer (list): List specifying the number of filters for each convolutional layer

    Returns:
        convolutional model with desired architecture

    Note:
        for each element in num_filters_per_layer:
            convolution (conv_kernel_size, num_filters, stride=1, padding="same") (use nn.Conv2d(..))
            relu (use nn.ReLU())
            max_pool(pool_kernel_size) (use nn.MaxPool2d(..))

        flatten layer (already given below)
        linear layer
        log softmax as final activation
    """
    assert (
        len(num_filters_per_layer) > 0
    ), "len(num_filters_per_layer) should be greater than 0"
    pool_kernel_size = 2
    conv_kernel_size = 3

    # OrderedDict is used to keep track of the order of the layers
    layers = OrderedDict()

    # START TODO ################
    input_channels = 1
    image_size = 28
    # create the first layer
    for i, num_filters in enumerate(num_filters_per_layer):
        layers["conv"+str(i+1)] = nn.Conv2d(input_channels, num_filters, conv_kernel_size,
                                            stride=1, padding="same")
        layers["relu"+str(i+1)] = nn.ReLU()
        layers["max_pool"+str(i+1)] = nn.MaxPool2d(pool_kernel_size)

        # set the input channels of the next layer to be the output channels of the current layer
        input_channels = num_filters

        # same padding and stride 1 for the conv layer leads to the same output size as the input size
        # get pool output size of maxpool layer
        image_size = image_size // pool_kernel_size

    conv_output_size = input_channels * image_size * image_size
    # END TODO ################

    layers["flatten"] = nn.Flatten()
    layers["linear"] = nn.Linear(conv_output_size, 10)
    layers["log_softmax"] = nn.LogSoftmax(dim=1)

    return nn.Sequential(layers)
