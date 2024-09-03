import torch
import torch.nn as nn
from typing import List


def pwl_mono_reg(model: nn.Module, x: torch.Tensor, monotonic_indices: List[int], offset: float = 0.) -> torch.Tensor:
    """
    Compute Point wise monotonicity regularization loss for a neural network.

    This function calculates a regularization term that encourages monotonicity
    in the specified input dimensions of the model's output.

    Args:
        model (nn.Module): The neural network model to regularize.
        x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        monotonic_indices (List[int]): Indices of input dimensions that should be monotonic.
        offset (float, optional): Offset value for the monotonicity constraint. Defaults to 0.

    Returns:
        torch.Tensor: The computed monotonicity regularization loss.
    """
    # Extract monotonic dimensions and enable gradient computation
    x_m = x[:, monotonic_indices]
    x_m.requires_grad_(True)
    # Prepare input for the model, replacing monotonic dimensions with gradients-enabled version
    x_grad = x.clone()
    x_grad[:, monotonic_indices] = x_m
    # Forward pass through the model
    y_pred_m = model(x_grad)
    # Compute gradients of the output with respect to monotonic inputs
    grads = torch.autograd.grad(y_pred_m.sum(), x_m, create_graph=True, allow_unused=True)[0]
    # Calculate divergence (sum of gradients across monotonic dimensions)
    divergence = grads.sum(dim=1)
    # Apply ReLU to penalize negative gradients (non-monotonic behavior)
    monotonicity_term = torch.relu(-divergence + offset)
    # Sum up the penalties to get the final regularization loss
    monotonicity_loss = monotonicity_term.sum()
    return monotonicity_loss
