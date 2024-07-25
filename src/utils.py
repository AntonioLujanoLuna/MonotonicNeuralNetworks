from typing import List, Literal, Union
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