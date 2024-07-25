import torch
import torch.nn as nn
from typing import List, Literal

from src.utils import init_weights


class StandardMLP(nn.Module):
    """
    Standard Multi-Layer Perceptron (MLP) implementation.

    This class provides a flexible MLP with customizable layer sizes, activation functions,
    and optional dropout.

    Attributes:
        input_size (int): Size of the input layer.
        hidden_sizes (List[int]): Sizes of the hidden layers.
        output_size (int): Size of the output layer.
        activation (nn.Module): Activation function used in hidden layers.
        output_activation (nn.Module): Activation function used in the output layer.
        dropout_rate (float): Dropout rate applied after each hidden layer.
        layers (nn.ModuleList): List of linear layers in the network.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        activation: nn.Module = nn.ReLU(),
        output_activation: nn.Module = nn.Identity(),
        dropout_rate: float = 0.0,
        init_method: Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'he_uniform', 'he_normal', 'truncated_normal'] = 'xavier_uniform'

    ):
        """
        Initialize the StandardMLP.

        Args:
            input_size (int): Size of the input layer.
            hidden_sizes (List[int]): List of sizes for the hidden layers.
            output_size (int): Size of the output layer.
            activation (nn.Module): Activation function for hidden layers.
            output_activation (nn.Module): Activation function for the output layer.
            dropout_rate (float): Dropout rate applied after each hidden layer.
            init_method (str): Weight initialization method.
        """
        super(StandardMLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation
        self.output_activation = output_activation
        self.dropout_rate = dropout_rate

        # Construct the layers
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i+1])
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

    def init_weights(self, method: Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'truncated_normal']) -> None:
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
            x = layer(x)
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
