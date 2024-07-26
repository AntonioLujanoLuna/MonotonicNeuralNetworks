import torch
import torch.nn as nn
import monotonicnetworks as lmn
from typing import List, Optional, Literal
from src.utils import init_weights

class LMNNetwork(nn.Module):
    """
    Lipschitz Monotonic Neural Network (LMNN) implementation.

    This class provides a flexible LMNN with customizable layer sizes, monotonicity constraints,
    Lipschitz constant, and output activation function.

    Attributes:
        input_size (int): Size of the input layer.
        hidden_sizes (List[int]): Sizes of the hidden layers.
        output_size (int): Size of the output layer.
        monotone_constraints (Optional[List[int]]): List of monotonicity constraints for each input feature.
        lipschitz_constant (float): Lipschitz constant for the network.
        sigma (float): Sigma value for the MonotonicWrapper.
        output_activation (nn.Module): Activation function used in the output layer.
        model (nn.Module): The underlying neural network model.
        wrapped_model (lmn.MonotonicWrapper): The model wrapped with MonotonicWrapper for monotonicity constraints.
    """

    def __init__(
            self,
            input_size: int,
            hidden_sizes: List[int],
            output_size: int = 1,
            monotone_constraints: Optional[List[int]] = None,
            lipschitz_constant: float = 1.0,
            sigma: float = 1.0,
            output_activation: nn.Module = nn.Identity(),
            init_method: Literal[
                'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'he_uniform', 'he_normal', 'truncated_normal'] = 'xavier_uniform'
    ):
        """
        Initialize the LMNNetwork.

        Args:
            input_size (int): Size of the input layer.
            hidden_sizes (List[int]): List of sizes for the hidden layers.
            output_size (int): Size of the output layer (default is 1).
            monotone_constraints (Optional[List[int]]): List of monotonicity constraints for each input feature.
                Use 1 for increasing, -1 for decreasing, and 0 for unrestricted. Default is None (all unrestricted).
            lipschitz_constant (float): Lipschitz constant for the network (default is 1.0).
            sigma (float): Sigma value for the MonotonicWrapper (default is 1.0).
            output_activation (nn.Module): Activation function for the output layer (default is nn.Identity()).
            init_method (str): Weight initialization method (default is 'xavier_uniform').
        """
        super(LMNNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.monotone_constraints = monotone_constraints
        self.lipschitz_constant = lipschitz_constant
        self.sigma = sigma
        self.output_activation = output_activation
        self.init_method = init_method
        self.model = self._build_model()
        self._init_weights(self.model)
        self.wrapped_model = lmn.MonotonicWrapper(
            self.model,
            lipschitz_const=self.lipschitz_constant,
            monotonic_constraints=self.monotone_constraints
        )

    def _build_model(self) -> nn.Sequential:
        """
        Build the underlying neural network model.

        Returns:
            nn.Sequential: The constructed neural network model.
        """
        layers = []
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]

        for i in range(len(layer_sizes) - 1):
            if i == 0:
                layers.append(lmn.direct_norm(nn.Linear(layer_sizes[i], layer_sizes[i + 1]), kind="one-inf"))
            else:
                layers.append(lmn.direct_norm(nn.Linear(layer_sizes[i], layer_sizes[i + 1]), kind="inf"))

            if i < len(layer_sizes) - 2:  # Don't add activation after the last layer
                layers.append(lmn.GroupSort(layer_sizes[i + 1] // 2))

        layers.append(self.output_activation)
        return nn.Sequential(*layers)

    def _init_weights(self, model: nn.Module) -> None:
        """
        Initialize the weights of the network.

        Args:
            model (nn.Module): The model whose weights are to be initialized.
        """
        for params in model.parameters():
            if len(params.shape) > 1:
                init_weights(params, method=self.init_method)
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
        return self.wrapped_model(x)

    def count_parameters(self) -> int:
        """
        Count the number of trainable parameters in the network.

        Returns:
            int: The total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def init_weights(self, method: Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'he_uniform', 'he_normal', 'truncated_normal']) -> None:
        """
        Initialize network parameters.

        Args:
            method (str): Weight initialization method.
        """
        self._init_weights(self.model)
        self.init_method = method