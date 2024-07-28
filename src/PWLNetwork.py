import torch
import torch.nn as nn
import torch.autograd as autograd
from typing import List, Literal
from MLP import StandardMLP
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


def custom_loss(output: torch.Tensor, target: torch.Tensor, model: StandardMLP, inputs: torch.Tensor, task_type: str,
                monotonic_indices: List[int], monotonicity_weight: float) -> torch.Tensor:
    if task_type == "regression":
        empirical_loss = nn.MSELoss()(output, target)
    else:
        empirical_loss = nn.BCELoss()(output, target)

    monotonic_loss = compute_monotonic_loss(model, inputs, monotonic_indices)
    return empirical_loss + monotonicity_weight * monotonic_loss


def compute_monotonic_loss(model: nn.Module, x: torch.Tensor, monotonic_indices: List[int]) -> torch.Tensor:
    monotonic_inputs = x[:, monotonic_indices]
    monotonic_inputs.requires_grad_(True)
    output = model(x)
    monotonic_gradients = autograd.grad(outputs=output, inputs=monotonic_inputs,
                                        grad_outputs=torch.ones_like(output),
                                        create_graph=True, retain_graph=True)[0]
    pwl = torch.sum(torch.max(torch.zeros_like(monotonic_gradients), -monotonic_gradients))
    return pwl

