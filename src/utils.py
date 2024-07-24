from typing import List
import numpy as np
import torch
from torch import nn


def monotonicity_check(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        data_x: torch.Tensor,
        monotonic_indices: List[int],
        device: torch.device
) -> float:
    model.train()
    optimizer.train()
    data_x = data_x.to(device)
    n_points = data_x.shape[0]

    if not monotonic_indices:
        print("Warning: No monotonic features specified. Skipping monotonicity check.")
        return 0.0

    # Separate monotonic and non-monotonic features
    monotonic_mask = torch.zeros(data_x.shape[1], dtype=torch.bool)
    monotonic_mask[monotonic_indices] = True
    data_monotonic = data_x[:, monotonic_mask]
    data_non_monotonic = data_x[:, ~monotonic_mask]

    data_monotonic.requires_grad_(True)

    def closure():
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(torch.cat([data_monotonic, data_non_monotonic], dim=1))
            loss = torch.sum(outputs)
        loss.backward()
        return loss

    optimizer.step(closure)

    grad_wrt_monotonic_input = data_monotonic.grad

    if grad_wrt_monotonic_input is None:
        print("Warning: Gradient is None. Check if the model is correctly set up for gradient computation.")
        return 0.0

    # Get component with minimum gradient for each point
    min_grad_wrt_monotonic_input = grad_wrt_monotonic_input.min(1)[0]

    # Count the points where gradients are negative
    number_of_non_monotonic_points = torch.sum(torch.where(min_grad_wrt_monotonic_input < 0, 1, 0)).item()

    return number_of_non_monotonic_points / n_points


def get_monotonic_indices(dataset_name: str) -> List[int]:
    # Remove the "load_" prefix if present
    dataset_name = dataset_name.replace("load_", "")

    # Define monotonic indices for each dataset
    monotonic_indices = {
        'abalone': [6, 7, 8, 9],
        'auto_mpg': [4, 5, 6],
        'blog_feedback': list(range(50, 59)),
        'boston_housing': [5],
        'compas': [0, 1, 2, 3],
        'era': [0, 1, 2, 3],
        'esl': [0, 1, 2, 3],
        'heart': [3, 4],
        'lev': [0, 1, 2, 3],
        'loan': [1, 4],
        'swd': [0, 1, 2, 4, 6, 8, 9]
    }
    return monotonic_indices.get(dataset_name, [])