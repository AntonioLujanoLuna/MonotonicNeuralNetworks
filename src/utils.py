from typing import List, Literal, Union, Dict
import torch
from torch import nn
import csv
import json
import itertools

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
        'abalone': [5, 6, 7, 8],
        'auto_mpg': [0, 1, 2, 3, 4, 5, 6],
        'blog_feedback': list(range(50, 59)),
        'boston_housing': [0, 5],
        'compas': [0, 1, 2, 3],
        'era': [0, 1, 2, 3],
        'esl': [0, 1, 2, 3],
        'heart': [3, 4],
        'lev': [0, 1, 2, 3],
        'loan': [0, 1, 2, 3, 4],
        'swd': [0, 1, 2, 4, 6, 8, 9]
    }
    return monotonic_indices.get(dataset_name, [])

def get_reordered_monotonic_indices(dataset_name: str) -> List[int]:
    # Remove the "load_" prefix if present
    dataset_name = dataset_name.replace("load_", "")

    # Define the number of monotonic features for each dataset
    monotonic_feature_counts = {
        'abalone': 4,
        'auto_mpg': 7,
        'blog_feedback': 9,
        'boston_housing': 2,
        'compas': 4,
        'era': 4,
        'esl': 4,
        'heart': 2,
        'lev': 4,
        'loan': 5,
        'swd': 7
    }

    # Get the number of monotonic features for the given dataset
    num_monotonic_features = monotonic_feature_counts.get(dataset_name, 0)

    # Return a list of consecutive integers starting from 0
    return list(range(num_monotonic_features))


def init_weights(module_or_tensor: Union[nn.Module, torch.Tensor],
                 method: Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'he_uniform', 'he_normal', 'truncated_normal', 'uniform', 'zeros'],
                 **kwargs) -> None:
    """
    Initialize weights of a module or tensor using the specified method.

    Args:
        module_or_tensor (Union[nn.Module, torch.Tensor]): The module or tensor to initialize.
        method (Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'he_uniform', 'he_normal', 'truncated_normal', 'uniform', 'zeros']): Initialization method.
        **kwargs: Additional arguments for specific initialization methods (e.g., mean, std, a, b).
    """
    def init_tensor(tensor):
        if method == 'xavier_uniform':
            nn.init.xavier_uniform_(tensor)
        elif method == 'xavier_normal':
            nn.init.xavier_normal_(tensor)
        elif method in ['kaiming_uniform', 'he_uniform']:
            nn.init.kaiming_uniform_(tensor)
        elif method in ['kaiming_normal', 'he_normal']:
            nn.init.kaiming_normal_(tensor)
        elif method == 'truncated_normal':
            mean = kwargs.get('mean', 0.)
            std = kwargs.get('std', 1.)
            with torch.no_grad():
                tensor.normal_(mean, std)
                while True:
                    cond = (tensor < mean - 2 * std) | (tensor > mean + 2 * std)
                    if not torch.sum(cond):
                        break
                    tensor[cond] = tensor[cond].normal_(mean, std)
        elif method == 'uniform':
            a = kwargs.get('a', 0.)
            b = kwargs.get('b', 1.)
            nn.init.uniform_(tensor, a=a, b=b)
        elif method == 'zeros':
            nn.init.zeros_(tensor)
        else:
            raise ValueError(f"Unsupported initialization method: {method}")

    if isinstance(module_or_tensor, nn.Module):
        for param in module_or_tensor.parameters():
            init_tensor(param)
    elif isinstance(module_or_tensor, torch.Tensor):
        init_tensor(module_or_tensor)
    else:
        raise TypeError("Input must be either nn.Module or torch.Tensor")

def transform_weights(weights: torch.Tensor, method: Literal['exp', 'explin', 'sqr']) -> torch.Tensor:
    """
    Apply the specified transformation to ensure positive weights.

    Args:
        weights (torch.Tensor): Input weights.
        method (Literal['exp', 'explin', 'sqr']): Transformation method.

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

def write_results_to_csv(filename: str, dataset_name: str, task_type: str, metric_name: str,
                         metric_value: float, metric_std: float, best_config: Dict, mono_metrics: Dict, n_params: int):
    # Convert best_config to a JSON string for easier CSV handling
    best_config_str = json.dumps(best_config)

    row = [
        dataset_name,
        task_type,
        metric_name,
        f"{metric_value:.4f}",
        f"{metric_std:.4f}",
        n_params,
        best_config_str
    ]

    # Add monotonicity metrics (mean and std) to the row
    for key in ['random', 'train', 'val']:
        mean, std = mono_metrics[key]
        row.extend([f"{mean:.4f}", f"{std:.4f}"])

    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def count_parameters(module: nn.Module) -> int:
    """
    Count the number of trainable parameters in a module.

    Args:
        module (nn.Module): The module to count parameters.

    Returns:
        int: The number of trainable parameters.
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def generate_layer_combinations(min_layers=1, max_layers=3, units=[8, 16, 32, 64]):
    """
    Generate all possible layer combinations for a feedforward neural network.

    Arguments
        min_layers (int): Minimum number of layers.
        max_layers (int): Maximum number of layers.
        units (List[int]): List of units per layer.

    Returns
        List[str]: List of layer combinations
    """
    combinations = []
    for n_layers in range(min_layers, max_layers + 1):
        for combo in itertools.product(units, repeat=n_layers):
            combinations.append(list(combo))
    return [str(combo) for combo in combinations]