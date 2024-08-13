import torch
import torch.nn as nn
import torch.autograd as autograd
from typing import List, Literal

from schedulefree import AdamWScheduleFree

from src.MLP import StandardMLP
from torch.utils.data import DataLoader

class PWLNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int,
                 monotonic_indices: List[int], monotonicity_weight: float = 1.0,
                 init_method: Literal[
                     'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'he_uniform', 'he_normal', 'truncated_normal'] = 'xavier_uniform'
                 ):
        """
        Initialize the PWLNetwork.

        Args:
            input_size (int): Size of the input layer.
            hidden_sizes (List[int]): List of sizes for the hidden layers.
            output_size (int): Size of the output layer.
            monotonic_indices (List[int]): Indices of monotonic features (assumed to be increasing).
            monotonicity_weight (float): Weight for the monotonicity loss term.
            init_method (str): Weight initialization method.
        """
        super(PWLNetwork, self).__init__()
        self.monotonic_indices = monotonic_indices
        self.monotonicity_weight = monotonicity_weight
        self.init_method = init_method

        self.network = StandardMLP(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            activation=nn.ReLU(),
            output_activation=nn.Identity(),
            init_method=self.init_method
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        return self.network(x)

    def compute_monotonic_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the monotonicity enforcing loss (PWL).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Monotonicity loss.
        """
        monotonic_inputs = x[:, self.monotonic_indices]
        monotonic_inputs.requires_grad_(True)
        output = self.forward(x)
        monotonic_gradients = autograd.grad(outputs=output, inputs=monotonic_inputs,
                                            grad_outputs=torch.ones_like(output),
                                            create_graph=True, retain_graph=True)[0]
        pwl = torch.sum(torch.max(torch.zeros_like(monotonic_gradients), -monotonic_gradients))
        return pwl

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor, loss_fn: nn.Module) -> torch.Tensor:
        """
        Compute the total loss including both empirical and monotonicity loss.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
            y (torch.Tensor): Target tensor of shape (batch_size, output_size).
            loss_fn (nn.Module): Loss function for empirical loss.

        Returns:
            torch.Tensor: Total loss.
        """
        y_pred = self.forward(x)
        empirical_loss = loss_fn(y_pred, y)
        monotonic_loss = self.compute_monotonic_loss(x)
        return empirical_loss + self.monotonicity_weight * monotonic_loss


def pwl(model: nn.Module, optimizer: AdamWScheduleFree, x: torch.Tensor, y: torch.Tensor,
        task_type: str, monotonic_indices: List[int], monotonicity_weight: float = 1.0, offset: float = 0.):
    criterion = nn.MSELoss() if task_type == "regression" else nn.BCELoss()

    def closure():
        optimizer.zero_grad()
        y_pred = model(x)
        empirical_loss = criterion(y_pred, y)  # This is L_NN

        monotonicity_loss = torch.tensor(0.0, device=x.device)
        if monotonic_indices:
            x_m = x[:, monotonic_indices]
            x_m.requires_grad_(True)
            # Create a new input tensor with gradients only for monotonic features
            x_grad = x.clone()
            x_grad[:, monotonic_indices] = x_m
            y_pred_m = model(x_grad)
            # Calculate gradients for each example with respect to monotonic features
            grads = torch.autograd.grad(y_pred_m.sum(), x_m, create_graph=True)[0]
            # Calculate divergence (sum of gradients across monotonic features)
            divergence = grads.sum(dim=1)
            # Apply max(0, -divergence + offset) for each example
            monotonicity_term = torch.relu(-divergence + offset)
            # Sum over all examples
            monotonicity_loss = monotonicity_term.sum()

        # Combine losses as per the equation
        total_loss = empirical_loss + monotonicity_weight * monotonicity_loss
        total_loss.backward()
        return total_loss

    loss = optimizer.step(closure)
    return loss