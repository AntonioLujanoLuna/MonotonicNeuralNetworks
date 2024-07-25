import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Literal
from utils import init_weights, transform_weights

class WeightsConstrainedMLP(nn.Module):
    """
    Multi-Layer Perceptron with constrained positive weights.

    This class implements a feedforward neural network where the weights
    are transformed to ensure they are always positive. It includes flexible
    activation functions and initialization methods.

    Attributes:
        layers (nn.ModuleList): List of linear layers in the network.
        transform (str): Type of transformation for ensuring positivity.
        activation (nn.Module): Activation function used in hidden layers.
        output_activation (nn.Module): Activation function used in the output layer.
        dropout_rate (float): Dropout rate applied after each hidden layer.
    """

    def __init__(
            self,
            input_size: int,
            hidden_sizes: List[int],
            output_size: int,
            activation: nn.Module = nn.ReLU(),
            output_activation: nn.Module = nn.Identity(),
            dropout_rate: float = 0.0,
            init_method: Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'he_uniform', 'he_normal', 'truncated_normal'] = 'xavier_uniform',
            transform: Literal['exp', 'explin', 'sqr'] = 'exp'
    ):
        """
        Initialize the WeightsConstrainedMLP.

        Args:
            input_size (int): Size of the input layer.
            hidden_sizes (List[int]): Sizes of the hidden layers.
            output_size (int): Size of the output layer.
            activation (nn.Module): Activation function for hidden layers.
            output_activation (nn.Module): Activation function for the output layer.
            dropout_rate (float): Dropout rate applied after each hidden layer.
            init_method (str): Weight initialization method.
            transform (str): Type of transformation for ensuring positivity.
        """
        super(WeightsConstrainedMLP, self).__init__()
        self.transform = transform
        self.activation = activation
        self.output_activation = output_activation
        self.dropout_rate = dropout_rate

        # Construct the layers
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            for i in range(len(layer_sizes) - 1)
        ])

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Initialize weights
        for params in self.parameters():
            if len(params.shape) > 1:
                init_weights(params, method=init_method)
            else:
                init_weights(params, method='zeros')

    def init_weights(self, method: Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'he_uniform', 'he_normal', 'truncated_normal']) -> None:
        """
        Initialize network parameters.

        Args:
            method (str): Weight initialization method.
        """
        for params in self.parameters():
            if len(params.shape) > 1:
                init_weights(params, method=method)
            else:
                init_weights(params, method='zeros')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        for i, layer in enumerate(self.layers):
            # Transform weights to ensure positivity
            positive_weights = transform_weights(layer.weight, self.transform)
            x = F.linear(x, positive_weights, layer.bias)

            if i < len(self.layers) - 1:  # Apply activation and dropout to all but the last layer
                x = self.activation(x)
                x = self.dropout(x)
            else:  # Apply output activation to the last layer
                x = self.output_activation(x)

        return x

    def count_parameters(self) -> int:
        """
        Count the number of trainable parameters in the network.

        Returns:
            int: The total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)