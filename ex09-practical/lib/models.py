"""CNN models to train"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet1(nn.Module):
    """
        The CNN model with 3 filters, kernel size 5, and padding 2
    """

    def __init__(self):
        super().__init__()
        # START TODO #############
        # initialize required parameters / layers needed to build the network
        self.filters_number = 3
        self.classes_number = 10

        self.conv = nn.Conv2d(in_channels=3, out_channels=self.filters_number, kernel_size=5, padding=2)
        self.max_pool = nn.MaxPool2d(kernel_size=2)

        # According to the assignment input images are 32x32, after applying one max_pool they will be 16x16
        # 10 is the number of classes for which we need to calculate the scores
        self.fc = nn.Linear(self.filters_number * 16 * 16, self.classes_number)
        # END TODO #############

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x: The input tensor with shape [batch_size, *feature_dim] (minibatch of data)
        Returns:
            scores: Pytorch tensor of shape (N, C) giving classification scores for x
        """
        # START TODO #############
        x = F.relu(self.conv(x))
        x = self.max_pool(x)

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        # END TODO #############
        return x


class ConvNet2(nn.Module):
    """
        The CNN model with 16 filters, kernel size 5, and padding 2
    """

    def __init__(self):
        super().__init__()
        # START TODO #############
        self.filters_number = 16
        self.classes_number = 10

        self.conv = nn.Conv2d(in_channels=3, out_channels=self.filters_number, kernel_size=5, padding=2)
        self.max_pool = nn.MaxPool2d(kernel_size=2)

        # According to the assignment input images are 32x32, after applying one max_pool they will be 16x16
        # 10 is the number of classes for which we need to calculate the scores
        self.fc = nn.Linear(self.filters_number * 16 * 16, self.classes_number)
        # END TODO #############

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x: The input tensor with shape (batch_size, *feature_dim)
            The input to the network will be a minibatch of data

        Returns:
            scores: PyTorch Tensor of shape (N, C) giving classification scores for x
        """
        # START TODO #############
        x = F.relu(self.conv(x))
        x = self.max_pool(x)

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        # END TODO #############
        return x


class ConvNet3(nn.Module):
    """
        The CNN model with 16 filters, kernel size 3, and padding 1
    """

    def __init__(self):
        super().__init__()
        # START TODO #############
        # Define the layers need to build the network
        self.filters_number = 16
        self.classes_number = 10

        self.conv = nn.Conv2d(in_channels=3, out_channels=self.filters_number, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2)

        # According to the assignment input images are 32x32, after applying one max_pool they will be 16x16
        # 10 is the number of classes for which we need to calculate the scores
        self.fc = nn.Linear(self.filters_number * 16 * 16, self.classes_number)
        # END TODO #############

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x: The input tensor with shape (batch_size, *feature_dim)
            The input to the network will be a minibatch of data

        Returns:
            scores: PyTorch Tensor of shape (N, C) giving classification scores for x
        """
        # START TODO #############
        x = F.relu(self.conv(x))
        x = self.max_pool(x)

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        # END TODO #############
        return x


class ConvNet4(nn.Module):
    """
        The CNN model with 16 filters, kernel size 3, padding 1 and batch normalization
    """

    def __init__(self):
        super().__init__()
        # START TODO #############
        # Define the layers need to build the network
        self.filters_number = 16
        self.classes_number = 10

        self.conv = nn.Conv2d(in_channels=3, out_channels=self.filters_number, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm2d(num_features=self.filters_number)
        self.max_pool = nn.MaxPool2d(kernel_size=2)

        # According to the assignment input images are 32x32, after applying one max_pool they will be 16x16
        # 10 is the number of classes for which we need to calculate the scores
        self.fc = nn.Linear(self.filters_number * 16 * 16, self.classes_number)
        # END TODO #############

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x: The input tensor with shape (batch_size, *feature_dim)
            The input to the network will be a minibatch of data

        Returns:
            scores: PyTorch Tensor of shape (N, C) giving classification scores for x
        """
        # START TODO #############
        x = self.conv(x)
        x = self.batch_norm(x)
        x = F.relu(x)

        x = self.max_pool(x)

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        # END TODO #############
        return x


class ConvNet5(nn.Module):
    """ Your custom CNN """

    def __init__(self):
        super().__init__()

        # START TODO #############
        self.classes_number = 10

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        self.batch_norm1 = nn.BatchNorm2d(num_features=32)
        self.batch_norm2 = nn.BatchNorm2d(num_features=64)
        self.batch_norm3 = nn.BatchNorm2d(num_features=128)

        self.max_pool = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, self.classes_number)
        # END TODO #############

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x: The input tensor with shape (batch_size, *feature_dim)
            The input to the network will be a minibatch of data

        Returns:
            scores: PyTorch Tensor of shape (N, C) giving classification scores for x
        """
        # START TODO #############
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.max_pool(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.max_pool(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = F.relu(x)
        x = self.max_pool(x)

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # END TODO #############
        return x
