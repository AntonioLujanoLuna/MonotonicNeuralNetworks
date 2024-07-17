import torch
import torch.nn as nn
from typing import List, Literal

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
        init_method: Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'truncated_normal'] = 'xavier_uniform'
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
        self.init_weights(init_method)

    def init_weights(self, method: Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'truncated_normal']) -> None:
        """
        Initialize network parameters.

        Args:
            method (str): Weight initialization method.
        """
        self._init_weights(self, method)

    @staticmethod
    def _init_weights(module: nn.Module, method: Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'truncated_normal']) -> None:
        """
        Initialize weights of a module using the specified method.

        Args:
            module (nn.Module): The module whose weights to initialize.
            method (str): Initialization method.
        """
        if method.startswith('xavier'):
            init_func = nn.init.xavier_uniform_ if method.endswith('uniform') else nn.init.xavier_normal_
        elif method.startswith('kaiming'):
            init_func = nn.init.kaiming_uniform_ if method.endswith('uniform') else nn.init.kaiming_normal_
        elif method == 'truncated_normal':
            def truncated_normal_(tensor, mean=0., std=1.):
                with torch.no_grad():
                    tensor.normal_(mean, std)
                    while True:
                        cond = (tensor < -2 * std) | (tensor > 2 * std)
                        if not torch.sum(cond):
                            break
                        tensor[cond] = tensor[cond].normal_(mean, std)
            init_func = truncated_normal_
        else:
            raise ValueError(f"Unsupported initialization method: {method}")

        for param in module.parameters():
            if len(param.shape) > 1:  # weights
                init_func(param)
            else:  # biases
                nn.init.zeros_(param)

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

    def check_grad(self) -> int:
        """
        Count the number of parameters with zero gradients.

        Returns:
            int: Number of parameters with zero gradients.
        """
        return sum(torch.sum((param.grad == 0).int()).item() for param in self.parameters() if param.grad is not None)

    def check_grad_neuron(self) -> int:
        """
        Count the number of neurons with all zero gradients.

        Returns:
            int: Number of neurons with all zero gradients.
        """
        return sum(self._check_grad_neuron(layer.weight, layer.bias) for layer in self.layers)

    @staticmethod
    def _check_grad_neuron(weights: torch.Tensor, biases: torch.Tensor) -> int:
        """
        Static method to count the number of neurons with all zero gradients.

        Args:
            weights (torch.Tensor): Weight tensor.
            biases (torch.Tensor): Bias tensor.

        Returns:
            int: Number of neurons with all zero gradients.
        """
        weights_zero = torch.all(weights.grad == 0, dim=1).int()
        biases_zero = (biases.grad == 0).int()
        return torch.sum(weights_zero * biases_zero).item()