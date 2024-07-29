import torch
import torch.nn as nn
import torch.autograd as autograd
from typing import List, Literal

from schedulefree import AdamWScheduleFree

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


def PWL_loss(output: torch.Tensor, target: torch.Tensor, model: StandardMLP, inputs: torch.Tensor, task_type: str,
                monotonic_indices: List[int], monotonicity_weight: float) -> torch.Tensor:
    if task_type == "regression":
        empirical_loss = nn.MSELoss()(output, target)
    else:
        empirical_loss = nn.BCELoss()(output, target)

    monotonic_loss = train_reg_term(model, inputs, monotonic_indices)
    return empirical_loss + monotonicity_weight * monotonic_loss


def train_reg_term(model: nn.Module, x: torch.Tensor, monotonic_indices: List[int]) -> torch.Tensor:
    monotonic_inputs = x[:, monotonic_indices]
    monotonic_inputs.requires_grad_(True)
    output = model(x)
    monotonic_gradients = autograd.grad(outputs=output, inputs=monotonic_inputs,
                                        grad_outputs=torch.ones_like(output),
                                        create_graph=True, retain_graph=True)[0]
    pwl = torch.sum(torch.max(torch.zeros_like(monotonic_gradients), -monotonic_gradients))
    return pwl


def pwl(model: nn.Module, optimizer: AdamWScheduleFree, x: torch.Tensor, y: torch.Tensor,
                     task_type: str, monotonic_indices: List[int], monotonicity_weight: float = 1.0,
                     b: float = 0.2) -> torch.Tensor:
    device = x.device

    # Compute empirical loss
    y_pred = model(x)
    if task_type == "regression":
        empirical_loss = nn.MSELoss()(y_pred, y)
    else:
        empirical_loss = nn.BCELoss()(y_pred, y)

    # Prepare for regularization
    model.train()
    optimizer.train()

    # Separate monotonic and non-monotonic features
    monotonic_mask = torch.zeros(x.shape[1], dtype=torch.bool)
    monotonic_mask[monotonic_indices] = True
    data_monotonic = x[:, monotonic_mask]
    data_non_monotonic = x[:, ~monotonic_mask]

    data_monotonic.requires_grad_(True)

    def closure():
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(torch.cat([data_monotonic, data_non_monotonic], dim=1))
            loss = torch.sum(outputs)
        loss.backward()
        return loss

    closure()
    grad_wrt_monotonic_input = data_monotonic.grad

    if grad_wrt_monotonic_input is None:
        print("Warning: Gradient is None. Check if the model is correctly set up for gradient computation.")
        return empirical_loss

    # Compute regularization (combines both approaches)
    grad_penalty = torch.relu(-grad_wrt_monotonic_input + b) ** 2
    regularization = torch.max(torch.sum(grad_penalty, dim=1))

    return empirical_loss + monotonicity_weight * regularization

