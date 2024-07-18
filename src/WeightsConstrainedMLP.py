import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Literal

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
        self.init_weights(init_method)

    def init_weights(self, method: Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'he_uniform', 'he_normal', 'truncated_normal']) -> None:
        """
        Initialize network parameters.

        Args:
            method (str): Weight initialization method.
        """
        self._init_weights(self, method)

    @staticmethod
    def _init_weights(module: nn.Module, method: Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'he_uniform', 'he_normal', 'truncated_normal']) -> None:
        """
        Initialize weights of a module using the specified method.

        Args:
            module (nn.Module): The module whose weights to initialize.
            method (str): Initialization method.
        """
        if method.startswith('xavier'):
            init_func = nn.init.xavier_uniform_ if method.endswith('uniform') else nn.init.xavier_normal_
        elif method.startswith('kaiming') or method.startswith('he'):
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

    @staticmethod
    def transform_weights(weights: torch.Tensor, method: Literal['exp', 'explin', 'sqr']) -> torch.Tensor:
        """
        Apply the specified transformation to ensure positive weights.

        Args:
            weights (torch.Tensor): Input weights.
            method (str): Transformation method.

        Returns:
            torch.Tensor: Transformed weights.
        """
        if method == 'exp':
            return torch.exp(weights)
        elif method == 'explin':
            return torch.where(weights > 1., weights, torch.exp(weights - 1.))
        elif method == 'sqr':
            return weights * weights
        else:
            raise ValueError(f"Unsupported transform method: {method}")

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
            positive_weights = self.transform_weights(layer.weight, self.transform)
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