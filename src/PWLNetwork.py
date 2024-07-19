import torch
import torch.nn as nn
import torch.autograd as autograd
from typing import List
from MLP import StandardMLP
from torch.utils.data import DataLoader

class PWLNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int,
                 monotonic_indices: List[int], monotonicity_weight: float = 1.0):
        """
        Initialize the PWLNetwork.

        Args:
            input_size (int): Size of the input layer.
            hidden_sizes (List[int]): List of sizes for the hidden layers.
            output_size (int): Size of the output layer.
            monotonic_indices (List[int]): Indices of monotonic features (assumed to be increasing).
            monotonicity_weight (float): Weight for the monotonicity loss term.
        """
        super(PWLNetwork, self).__init__()
        self.monotonic_indices = monotonic_indices
        self.monotonicity_weight = monotonicity_weight

        self.network = StandardMLP(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            activation=nn.ReLU(),
            output_activation=nn.Identity()
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

def train_pwl_network(model: PWLNetwork, train_loader: DataLoader,
                      optimizer: torch.optim.Optimizer, loss_fn: nn.Module, num_epochs: int):
    """
    Train the PWLNetwork.

    Args:
        model (PWLNetwork): The PWLNetwork model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        loss_fn (nn.Module): Loss function for empirical loss.
        num_epochs (int): Number of epochs to train for.
    """
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = model.compute_loss(batch_x, batch_y, loss_fn)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")

# Example usage:
# input_size = 10
# hidden_sizes = [64, 32]
# output_size = 1
# monotonic_indices = [0, 2]  # Features 0 and 2 are monotonically increasing
# model = PWLNetwork(input_size, hidden_sizes, output_size, monotonic_indices)
# optimizer = torch.optim.Adam(model.parameters())
# loss_fn = nn.MSELoss()
# train_loader = torch.utils.data.DataLoader(your_dataset, batch_size=32, shuffle=True)
# train_pwl_network(model, train_loader, optimizer, loss_fn, num_epochs=100)