import torch
import torch.nn as nn
from typing import List, Literal

from src.utils import init_weights, transform_weights


class MinMaxNetworkBase(nn.Module):
    """
    Base class for MinMaxNetwork implementations.

    This class provides common functionality for various MinMaxNetwork variants,
    including weight initialization, gradient checking, and weight transformation.

    Attributes:
        K (int): Number of groups.
        h_K (int): Number of neurons per group.
        monotonic_mask (torch.Tensor): Boolean mask for monotonic features.
        transform (str): Type of transformation for ensuring positivity.
        use_sigmoid (bool): Whether to apply sigmoid to the output.
        z (nn.ParameterList): List of weight parameters.
        t (nn.ParameterList): List of bias parameters.
    """

    def __init__(
        self,
        n: int,
        K: int,
        h_K: int,
        monotonic_indices: List[int],
        init_method: Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'he_uniform', 'he_normal', 'truncated_normal'] = 'xavier_uniform',
        transform: Literal['exp', 'explin', 'sqr'] = 'exp',
        use_sigmoid: bool = False
    ) -> None:
        """
        Initialize the MinMaxNetworkBase.

        Args:
            n (int): Number of inputs.
            K (int): Number of groups.
            h_K (int): Number of neurons per group.
            monotonic_indices (List[int]): Indices of monotonic features.
            init_method (str): Weight initialization method.
            transform (str): Type of transformation for ensuring positivity.
            use_sigmoid (bool): Whether to apply sigmoid to the output.
        """
        super(MinMaxNetworkBase, self).__init__()
        self.K = K
        self.h_K = h_K
        self.monotonic_mask = torch.zeros(n, dtype=torch.bool)
        self.monotonic_mask[monotonic_indices] = True
        self.transform = transform
        self.use_sigmoid = use_sigmoid

        self.z = nn.ParameterList([nn.Parameter(torch.empty(h_K, n)) for _ in range(K)])
        self.t = nn.ParameterList([nn.Parameter(torch.empty(h_K)) for _ in range(K)])

        for params in self.parameters():
            if len(params.shape) > 1:
                init_weights(params, method=init_method)
            else:
                init_weights(params, method='zeros')

    def init_weights(self, method: Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'he_uniform', 'he_normal', 'truncated_normal']) -> None:
        """
        Initialize network parameters.

        Args:
            method (str): Weight initialization method.
        """
        for params in self.parameters():
            if len(params.shape) > 1:
                init_weights(params, method=method)
            else:
                init_weights(params, method='zeros')

    def transform_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Apply the specified transformation to ensure positive weights.

        Args:
            weights (torch.Tensor): Input weights.

        Returns:
            torch.Tensor: Transformed weights.
        """
        if self.transform == 'exp':
            return torch.exp(weights)
        elif self.transform == 'explin':
            return torch.where(weights > 1., weights, torch.exp(weights - 1.))
        elif self.transform == 'sqr':
            return weights * weights
        else:
            raise ValueError(f"Unsupported transform method: {self.transform}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size,).
        """
        group_outputs = []
        for i in range(self.K):
            w = torch.where(self.monotonic_mask, transform_weights(self.z[i]), self.z[i])
            a = torch.matmul(x, w.t()) + self.t[i]
            g = self.group_operation(a)
            group_outputs.append(g)

        y = self.combine_outputs(group_outputs)
        return torch.sigmoid(y) if self.use_sigmoid else y

    def group_operation(self, a: torch.Tensor) -> torch.Tensor:
        """
        Perform the group operation on the input tensor.

        Args:
            a (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Result of the group operation.
        """
        raise NotImplementedError("Subclasses must implement group_operation method")

    def combine_outputs(self, group_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Combine the outputs from all groups.

        Args:
            group_outputs (List[torch.Tensor]): List of outputs from each group.

        Returns:
            torch.Tensor: Combined output.
        """
        raise NotImplementedError("Subclasses must implement combine_outputs method")

    def count_parameters(self) -> int:
        """
        Count the number of trainable parameters in the network.

        Returns:
            int: The total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MinMaxNetwork(MinMaxNetworkBase):
    """
    MinMaxNetwork implementation with mask for non-monotonic features.

    This class extends MinMaxNetworkBase and implements the specific group
    operation (max) and output combination (min) for the MinMax network.
    """

    def group_operation(self, a: torch.Tensor) -> torch.Tensor:
        """
        Perform the max operation for each group.

        Args:
            a (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Maximum values along the second dimension.
        """
        return torch.max(a, dim=1)[0]

    def combine_outputs(self, group_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Combine the group outputs using the min operation.

        Args:
            group_outputs (List[torch.Tensor]): List of outputs from each group.

        Returns:
            torch.Tensor: Minimum values across all groups.
        """
        return torch.min(torch.stack(group_outputs), dim=0)[0]


class SmoothMinMaxNetwork(MinMaxNetworkBase):
    """
    SmoothMinMaxNetwork implementation with mask for non-monotonic features.

    This class extends MinMaxNetworkBase and implements the specific group
    operation (soft max) and output combination (soft min) for the Smooth MinMax network.
    """

    def __init__(self, *args, beta: float = -1.0, **kwargs) -> None:
        """
        Initialize the SmoothMinMaxNetwork.

        Args:
            *args: Variable length argument list.
            beta (float): Initial value for the smoothing parameter.
            **kwargs: Arbitrary keyword arguments.
        """
        super(SmoothMinMaxNetwork, self).__init__(*args, **kwargs)
        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float))

    def soft_max(self, a: torch.Tensor) -> torch.Tensor:
        """
        Compute the soft maximum.

        Args:
            a (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Soft maximum of the input tensor.
        """
        return torch.logsumexp(torch.exp(self.beta) * a, dim=1) / torch.exp(self.beta)

    def soft_min(self, a: torch.Tensor) -> torch.Tensor:
        """
        Compute the soft minimum.

        Args:
            a (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Soft minimum of the input tensor.
        """
        return -torch.logsumexp(-torch.exp(self.beta) * a, dim=1) / torch.exp(self.beta)

    def group_operation(self, a: torch.Tensor) -> torch.Tensor:
        """
        Perform the soft max operation for each group.

        Args:
            a (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Soft maximum values along the second dimension.
        """
        return self.soft_max(a)

    def combine_outputs(self, group_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Combine the group outputs using the soft min operation.

        Args:
            group_outputs (List[torch.Tensor]): List of outputs from each group.

        Returns:
            torch.Tensor: Soft minimum values across all groups.
        """
        return self.soft_min(torch.stack(group_outputs, dim=1))


import torch
import torch.nn as nn
from typing import List, Literal, Union, Tuple, Callable


# Assuming MinMaxNetworkBase, MinMaxNetwork, and SmoothMinMaxNetwork are already defined as in the previous artifact

class MLPMixin:
    """
    Mixin class that adds an auxiliary MLP to MinMaxNetwork variants.

    This mixin provides functionality to process non-monotonic inputs
    through an additional Multi-Layer Perceptron (MLP) network.
    """

    def __init__(self, n: int, aux_hidden_units: int = 64,
                 aux_activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU(),
                 init_method: str = 'xavier_uniform') -> None:
        """
        Initialize the MLPMixin.

        Args:
            n (int): Total number of input features.
            aux_hidden_units (int): Number of hidden units in the auxiliary network.
            aux_activation (Callable[[torch.Tensor], torch.Tensor]): Activation function for the auxiliary network.
            init_method (str): Weight initialization method for the auxiliary network.
        """
        # Ensure self.monotonic_mask is defined in the main class
        non_monotonic_dim = int(sum(~self.monotonic_mask))
        self.auxiliary_net = nn.Sequential(
            nn.Linear(non_monotonic_dim, aux_hidden_units),
            aux_activation,
            nn.Linear(aux_hidden_units, 1)
        )
        for params in self.auxiliary_net.parameters():
            if len(params.shape) > 1:
                init_weights(params, method=init_method)
            else:
                init_weights(params, method='zeros')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network with auxiliary MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size,).
        """
        x_unconstrained = x[:, ~self.monotonic_mask]
        aux_output = self.auxiliary_net(x_unconstrained)

        group_outputs = []
        for i in range(self.K):
            w = torch.where(self.monotonic_mask, self.transform_weights(self.z[i]), self.z[i])
            a = torch.matmul(x, w.t()) + self.t[i] + aux_output.squeeze(-1)
            g = self.group_operation(a)
            group_outputs.append(g)

        y = self.combine_outputs(group_outputs)
        return torch.sigmoid(y) if self.use_sigmoid else y

    def check_grad(self) -> int:
        """
        Count the number of parameters with zero gradients, including the auxiliary network.

        Returns:
            int: Number of parameters with zero gradients.
        """
        base_grad = super().check_grad()
        aux_grad = sum(torch.sum((param.grad == 0).int()).item() for param in self.auxiliary_net.parameters() if
                       param.grad is not None)
        return base_grad + aux_grad

    def count_parameters(self) -> int:
        """
        Count the number of trainable parameters in the network, including the auxiliary network.

        Returns:
            int: The total number of trainable parameters.
        """
        base_params = super().count_parameters()
        aux_params = sum(p.numel() for p in self.auxiliary_net.parameters() if p.requires_grad)
        return base_params + aux_params


class MinMaxNetworkWithMLP(MLPMixin, MinMaxNetwork):
    """
    MinMaxNetwork with auxiliary MLP for partially monotone problems.

    This class combines the MinMaxNetwork with an auxiliary MLP to handle
    non-monotonic inputs separately.
    """

    def __init__(self, n: int, K: int, h_K: int, monotonic_indices: List[int],
                 aux_hidden_units: int = 64,
                 init_method: Literal[
                     'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal'] = 'xavier_uniform',
                 transform: Literal['exp', 'explin', 'sqr'] = 'exp',
                 use_sigmoid: bool = False,
                 aux_activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()) -> None:
        """
        Initialize the MinMaxNetworkWithMLP.

        Args:
            n (int): Number of inputs.
            K (int): Number of groups.
            h_K (int): Number of neurons per group.
            monotonic_indices (List[int]): Indices of monotonic features.
            aux_hidden_units (int): Number of hidden units in the auxiliary network.
            init_method (str): Weight initialization method.
            transform (str): Type of transformation for ensuring positivity.
            use_sigmoid (bool): Whether to apply sigmoid to the output.
            aux_activation (Callable[[torch.Tensor], torch.Tensor]): Activation function for the auxiliary network.
        """
        MinMaxNetwork.__init__(self, n, K, h_K, monotonic_indices, init_method, transform, use_sigmoid)
        MLPMixin.__init__(self, n, aux_hidden_units, aux_activation, init_method)


class SmoothMinMaxNetworkWithMLP(MLPMixin, SmoothMinMaxNetwork):
    """
    SmoothMinMaxNetwork with auxiliary MLP for partially monotone problems.

    This class combines the SmoothMinMaxNetwork with an auxiliary MLP to handle
    non-monotonic inputs separately.
    """

    def __init__(self, n: int, K: int, h_K: int, monotonic_indices: List[int],
                 aux_hidden_units: int = 64, beta: float = -1.0,
                 init_method: Literal[
                     'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal'] = 'xavier_uniform',
                 transform: Literal['exp', 'explin', 'sqr'] = 'exp',
                 use_sigmoid: bool = False,
                 aux_activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()) -> None:
        """
        Initialize the SmoothMinMaxNetworkWithMLP.

        Args:
            n (int): Number of inputs.
            K (int): Number of groups.
            h_K (int): Number of neurons per group.
            monotonic_indices (List[int]): Indices of monotonic features.
            aux_hidden_units (int): Number of hidden units in the auxiliary network.
            beta (float): Initial value for the smoothing parameter.
            init_method (str): Weight initialization method.
            transform (str): Type of transformation for ensuring positivity.
            use_sigmoid (bool): Whether to apply sigmoid to the output.
            aux_activation (Callable[[torch.Tensor], torch.Tensor]): Activation function for the auxiliary network.
        """
        SmoothMinMaxNetwork.__init__(self, n, K, h_K, monotonic_indices, init_method, transform, use_sigmoid, beta=beta)
        MLPMixin.__init__(self, n, aux_hidden_units, aux_activation, init_method)
