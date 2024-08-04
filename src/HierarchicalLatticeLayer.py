import torch
import torch.nn as nn
from pmlayer.torch.layers import HLattice
from typing import List, Optional, Literal
from src.MLP import StandardMLP


class HLLNetwork(nn.Module):
    """
    Hybrid Lattice Layer (HLL) Network implementation using pmlayer and StandardMLP.

    This class provides a flexible HLL network with customizable lattice sizes,
    increasing dimensions, and MLP architecture.

    Attributes:
        dim (int): Total number of input dimensions.
        lattice_sizes (List[int]): Sizes of each lattice dimension.
        increasing (List[int]): Indices of input dimensions that should be increasing.
        mlp_neurons (List[int]): Sizes of hidden layers in the MLP.
        activation (nn.Module): Activation function for the MLP.
        dropout_rate (float): Dropout rate for the MLP.
        output_activation (nn.Module): Activation function for the output layer.
        init_method (str): Weight initialization method for the MLP.
        model (HLattice): The underlying HLattice model.
    """

    def __init__(
            self,
            dim: int,
            lattice_sizes: List[int],
            increasing: List[int],
            mlp_neurons: List[int],
            device: torch.device,
            activation: nn.Module = nn.ReLU(),
            dropout_rate: float = 0.0,
            output_activation: Optional[nn.Module] = None,
            init_method: Literal[
                'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'truncated_normal'] = 'xavier_uniform'
    ):
        """
        Initialize the HLLNetwork.

        Args:
            dim (int): Total number of input dimensions.
            lattice_sizes (List[int]): Sizes of each lattice dimension.
            increasing (List[int]): Indices of input dimensions that should be increasing.
            mlp_neurons (List[int]): Sizes of hidden layers in the MLP.
            activation (nn.Module): Activation function for the MLP.
            dropout_rate (float): Dropout rate for the MLP.
            device (torch.device): Device
            init_method (str): Weight initialization method for the MLP.
        """
        super(HLLNetwork, self).__init__()
        self.dim = dim
        self.device = device
        self.lattice_sizes = lattice_sizes
        self.increasing = increasing
        self.mlp_neurons = mlp_neurons
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.output_activation = output_activation
        self.init_method = init_method
        self.model = self._build_model()

    def _build_model(self) -> HLattice:
        """
        Build the underlying HLattice model.

        Returns:
            HLattice: The constructed HLattice model.
        """
        input_len = self.dim - len(self.increasing)
        output_len = torch.prod(torch.tensor(self.lattice_sizes)).item()

        ann = StandardMLP(
            input_size=input_len,
            hidden_sizes=self.mlp_neurons,
            output_size=output_len,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            init_method=self.init_method,
            output_activation=self.output_activation
        )
        return HLattice(self.dim, torch.tensor(self.lattice_sizes), self.increasing, ann)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        return self.model(x)


    def count_parameters(self) -> int:
        """
        Count the number of trainable parameters in the network.

        Returns:
            int: The total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Example usage
"""
dim = 10
lattice_sizes = [3, 3, 3, 3]  # Example lattice sizes
increasing = [0, 1]  # Example increasing dimensions
mlp_neurons = [64, 32]  # Example MLP architecture

model = HLLNetwork(
    dim=dim,
    lattice_sizes=lattice_sizes,
    increasing=increasing,
    mlp_neurons=mlp_neurons,
    activation=nn.ReLU(),
    dropout_rate=0.1,
    init_method='xavier_uniform'
)

# Assuming you have your data as torch tensors
history = model.fit(x_train, y_train, x_val, y_val, epochs=1000, learning_rate=0.01)

# Make predictions
predictions = model.predict(x_test)
"""