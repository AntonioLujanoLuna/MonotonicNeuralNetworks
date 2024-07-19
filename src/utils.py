import torch
import torch.nn as nn
from typing import List
from torch.utils.data import DataLoader, TensorDataset


def compute_monotonicity_metric(model: nn.Module,
                                data_x: torch.Tensor,
                                monotonic_indices: List[int],
                                sample_size: int = 10000,
                                data_type: str = 'random') -> float:
    """
    Compute the monotonicity metric ρ as described in the image.

    Args:
        model (nn.Module): The model to evaluate.
        data_x (torch.Tensor): Input data tensor.
        monotonic_indices (List[int]): Indices of features expected to be monotonic.
        sample_size (int): Number of points to sample (for 'random' type).
        data_type (str): One of 'random', 'train', or 'test'.

    Returns:
        float: The monotonicity metric ρ.
    """
    model.eval()
    device = next(model.parameters()).device

    if data_type == 'random':
        x = torch.rand(sample_size, data_x.shape[1], device=device)
    elif data_type in ['train', 'test']:
        if sample_size < data_x.shape[0]:
            indices = torch.randperm(data_x.shape[0])[:sample_size]
            x = data_x[indices].to(device)
        else:
            x = data_x.to(device)
    else:
        raise ValueError("data_type must be one of 'random', 'train', or 'test'")

    x_monotonic = x[:, monotonic_indices]
    x_monotonic.requires_grad_(True)

    y_pred = model(x)

    grads = torch.autograd.grad(y_pred.sum(), x_monotonic, create_graph=True)[0]
    min_grads = grads.min(dim=1)[0]

    non_monotonic_points = (min_grads < 0).float().sum().item()
    rho = 1 - (non_monotonic_points / x.shape[0])

    return rho

# Example usage:
# model = YourModel()
# train_data = YourTrainData()
# test_data = YourTestData()
# monotonic_indices = [0, 1, 2]  # Assuming features 0, 1, and 2 should be monotonic

# rho_random = compute_monotonicity_metric(model, train_data, monotonic_indices, data_type='random')
# rho_train = compute_monotonicity_metric(model, train_data, monotonic_indices, data_type='train')
# rho_test = compute_monotonicity_metric(model, test_data, monotonic_indices, data_type='test')

# print(f"ρ_random: {rho_random}")
# print(f"ρ_train: {rho_train}")
# print(f"ρ_test: {rho_test}")