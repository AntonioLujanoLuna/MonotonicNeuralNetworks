import torch
import torch.nn as nn
from typing import Callable, List, Literal

def init_weights(module: nn.Module, method: Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal']) -> None:
    """
    Initialize weights of a module using the specified method.

    Args:
        module (nn.Module): The module whose weights to initialize.
        method (Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal']): Initialization method.
    """
    if method.startswith('xavier'):
        gain = nn.init.calculate_gain('tanh')
        init_func = nn.init.xavier_uniform_ if method.endswith('uniform') else nn.init.xavier_normal_
    elif method.startswith('kaiming'):
        init_func = nn.init.kaiming_uniform_ if method.endswith('uniform') else nn.init.kaiming_normal_
        gain = 1.0  # Kaiming initialization already accounts for gain
    else:
        raise ValueError(f"Unsupported initialization method: {method}")

    for param in module.parameters():
        if len(param.shape) > 1:  # weights
            init_func(param, gain=gain)
        else:  # biases
            nn.init.zeros_(param)

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

def check_grad(module: nn.Module) -> int:
    """
    Count the number of parameters with zero gradients.

    Args:
        module (nn.Module): The module to check.

    Returns:
        int: Number of parameters with zero gradients.
    """
    return sum(torch.sum((param.grad == 0).int()).item() for param in module.parameters() if param.grad is not None)

def check_grad_neuron(weights: torch.Tensor, biases: torch.Tensor) -> int:
    """
    Count the number of neurons with all zero gradients.

    Args:
        weights (torch.Tensor): Weight tensor.
        biases (torch.Tensor): Bias tensor.

    Returns:
        int: Number of neurons with all zero gradients.
    """
    weights_zero = torch.all(weights.grad == 0, dim=1).int()
    biases_zero = (biases.grad == 0).int()
    return torch.sum(weights_zero * biases_zero).item()

class MinMaxNetwork(nn.Module):
    def __init__(
        self,
        n: int,
        K: int,
        h_K: int,
        monotonic_indices: List[int],
        init_method: Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal'] = 'xavier_uniform',
        transform: Literal['exp', 'explin', 'sqr'] = 'exp',
        use_sigmoid: bool = False
    ):
        """
        MinMaxNetwork implementation with mask for non-monotonic features.

        Args:
            n (int): Number of inputs.
            K (int): Number of groups.
            h_K (int): Number of neurons per group.
            monotonic_indices (List[int]): Indices of monotonic features.
            init_method (Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal']): Weight initialization method.
            transform (Literal['exp', 'explin', 'sqr']): Type of transformation for ensuring positivity.
            use_sigmoid (bool): Whether to apply sigmoid to the output.
        """
        super(MinMaxNetwork, self).__init__()
        self.K = K
        self.h_K = h_K
        self.monotonic_mask = torch.zeros(n, dtype=torch.bool)
        self.monotonic_mask[monotonic_indices] = True
        self.transform = transform
        self.use_sigmoid = use_sigmoid

        self.z = nn.ParameterList([nn.Parameter(torch.empty(h_K, n)) for _ in range(K)])
        self.t = nn.ParameterList([nn.Parameter(torch.empty(h_K)) for _ in range(K)])

        self.init_weights(init_method)

    def init_weights(self, method: Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal']) -> None:
        """Initialize network parameters."""
        init_weights(self, method)

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
            w = torch.where(self.monotonic_mask, transform_weights(self.z[i], self.transform), self.z[i])
            a = torch.matmul(x, w.t()) + self.t[i]
            g, _ = torch.max(a, dim=1)
            group_outputs.append(g)

        y = torch.min(torch.stack(group_outputs), dim=0)[0]
        return torch.sigmoid(y) if self.use_sigmoid else y

    def check_grad(self) -> int:
        """
        Count the number of parameters with zero gradients.

        Returns:
            int: Number of parameters with zero gradients.
        """
        return check_grad(self)

    def check_grad_neuron(self) -> int:
        """
        Count the number of neurons with all zero gradients.

        Returns:
            int: Number of neurons with all zero gradients.
        """
        return sum(check_grad_neuron(self.z[i], self.t[i]) for i in range(self.K))

class SmoothMinMaxNetwork(nn.Module):
    def __init__(
        self,
        n: int,
        K: int,
        h_K: int,
        monotonic_indices: List[int],
        beta: float = -1.0,
        init_method: Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal'] = 'xavier_uniform',
        transform: Literal['exp', 'explin', 'sqr'] = 'exp',
        use_sigmoid: bool = False
    ):
        """
        SmoothMinMaxNetwork implementation with mask for non-monotonic features.

        Args:
            n (int): Number of inputs.
            K (int): Number of groups.
            h_K (int): Number of neurons per group.
            monotonic_indices (List[int]): Indices of monotonic features.
            beta (float): Initial value for the smoothing parameter.
            init_method (Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal']): Weight initialization method.
            transform (Literal['exp', 'explin', 'sqr']): Type of transformation for ensuring positivity.
            use_sigmoid (bool): Whether to apply sigmoid to the output.
        """
        super(SmoothMinMaxNetwork, self).__init__()
        self.K = K
        self.h_K = h_K
        self.monotonic_mask = torch.zeros(n, dtype=torch.bool)
        self.monotonic_mask[monotonic_indices] = True
        self.transform = transform
        self.use_sigmoid = use_sigmoid

        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float))
        self.z = nn.ParameterList([nn.Parameter(torch.empty(h_K, n)) for _ in range(K)])
        self.t = nn.ParameterList([nn.Parameter(torch.empty(h_K)) for _ in range(K)])

        self.init_weights(init_method)

    def init_weights(self, method: Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal']) -> None:
        """Initialize network parameters."""
        init_weights(self, method)

    def soft_max(self, a: torch.Tensor) -> torch.Tensor:
        """Compute the soft maximum."""
        return torch.logsumexp(torch.exp(self.beta) * a, dim=1) / torch.exp(self.beta)

    def soft_min(self, a: torch.Tensor) -> torch.Tensor:
        """Compute the soft minimum."""
        return -torch.logsumexp(-torch.exp(self.beta) * a, dim=1) / torch.exp(self.beta)

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
            w = torch.where(self.monotonic_mask, transform_weights(self.z[i], self.transform), self.z[i])
            a = torch.matmul(x, w.t()) + self.t[i]
            g = self.soft_max(a)
            group_outputs.append(g)

        y = self.soft_min(torch.stack(group_outputs, dim=1))
        return torch.sigmoid(y) if self.use_sigmoid else y

    def check_grad(self) -> int:
        """
        Count the number of parameters with zero gradients.

        Returns:
            int: Number of parameters with zero gradients.
        """
        return check_grad(self)

    def check_grad_neuron(self) -> int:
        """
        Count the number of neurons with all zero gradients.

        Returns:
            int: Number of neurons with all zero gradients.
        """
        return sum(check_grad_neuron(self.z[i], self.t[i]) for i in range(self.K))

class MinMaxNetworkWithMLP(nn.Module):
    def __init__(
        self,
        n: int,
        K: int,
        h_K: int,
        monotonic_indices: List[int],
        aux_hidden_units: int = 64,
        init_method: Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal'] = 'xavier_uniform',
        transform: Literal['exp', 'explin', 'sqr'] = 'exp',
        use_sigmoid: bool = False,
        aux_activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()
    ):
        """
        MinMaxNetwork with auxiliary MLP for partially monotone problems.

        Args:
            n (int): Number of inputs.
            K (int): Number of groups.
            h_K (int): Number of neurons per group.
            monotonic_indices (List[int]): Indices of monotonic features.
            aux_hidden_units (int): Number of hidden units in the auxiliary network.
            init_method (Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal']): Weight initialization method.
            transform (Literal['exp', 'explin', 'sqr']): Type of transformation for ensuring positivity.
            use_sigmoid (bool): Whether to apply sigmoid to the output.
            aux_activation (Callable[[torch.Tensor], torch.Tensor]): Activation function for the auxiliary network.
        """
        super(MinMaxNetworkWithMLP, self).__init__()
        self.K = K
        self.h_K = h_K
        self.monotonic_mask = torch.zeros(n, dtype=torch.bool)
        self.monotonic_mask[monotonic_indices] = True
        self.transform = transform
        self.use_sigmoid = use_sigmoid

        self.z = nn.ParameterList([nn.Parameter(torch.empty(h_K, n)) for _ in range(K)])
        self.t = nn.ParameterList([nn.Parameter(torch.empty(h_K)) for _ in range(K)])

        # Auxiliary network for unconstrained inputs
        non_monotonic_dim = n - len(monotonic_indices)
        self.auxiliary_net = nn.Sequential(
            nn.Linear(non_monotonic_dim, aux_hidden_units),
            aux_activation,
            nn.Linear(aux_hidden_units, 1)
        )

        self.init_weights(init_method)

    def init_weights(self, method: Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal']) -> None:
        """Initialize network parameters."""
        init_weights(self, method)
        init_weights(self.auxiliary_net, method)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size,).
        """
        x_unconstrained = x[:, ~self.monotonic_mask]
        aux_output = self.auxiliary_net(x_unconstrained)

        group_outputs = []
        for i in range(self.K):
            w = torch.where(self.monotonic_mask, transform_weights(self.z[i], self.transform), self.z[i])
            a = torch.matmul(x, w.t()) + self.t[i] + aux_output.squeeze(-1)
            g, _ = torch.max(a, dim=1)
            group_outputs.append(g)

        y = torch.min(torch.stack(group_outputs, dim=1), dim=1)[0]
        return torch.sigmoid(y) if self.use_sigmoid else y

    def check_grad(self) -> int:
        """
        Count the number of parameters with zero gradients.

        Returns:
            int: Number of parameters with zero gradients.
        """
        return check_grad(self) + check_grad(self.auxiliary_net)

    def check_grad_neuron(self) -> int:
        """
        Count the number of neurons with all zero gradients.

        Returns:
            int: Number of neurons with all zero gradients.
        """
        return sum(check_grad_neuron(self.z[i], self.t[i]) for i in range(self.K))

class SmoothMinMaxNetworkWithMLP(nn.Module):
    def __init__(
        self,
        n: int,
        K: int,
        h_K: int,
        monotonic_indices: List[int],
        aux_hidden_units: int = 64,
        beta: float = -1.0,
        init_method: Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal'] = 'xavier_uniform',
        transform: Literal['exp', 'explin', 'sqr'] = 'exp',
        use_sigmoid: bool = False,
        aux_activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()
    ):
        """
        SmoothMinMaxNetwork with auxiliary MLP for partially monotone problems.

        Args:
            n (int): Number of inputs.
            K (int): Number of groups.
            h_K (int): Number of neurons per group.
            monotonic_indices (List[int]): Indices of monotonic features.
            aux_hidden_units (int): Number of hidden units in the auxiliary network.
            beta (float): Initial value for the smoothing parameter.
            init_method (Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal']): Weight initialization method.
            transform (Literal['exp', 'explin', 'sqr']): Type of transformation for ensuring positivity.
            use_sigmoid (bool): Whether to apply sigmoid to the output.
            aux_activation (Callable[[torch.Tensor], torch.Tensor]): Activation function for the auxiliary network.
        """
        super(SmoothMinMaxNetworkWithMLP, self).__init__()
        self.K = K
        self.h_K = h_K
        self.monotonic_mask = torch.zeros(n, dtype=torch.bool)
        self.monotonic_mask[monotonic_indices] = True
        self.transform = transform
        self.use_sigmoid = use_sigmoid

        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float))
        self.z = nn.ParameterList([nn.Parameter(torch.empty(h_K, n)) for _ in range(K)])
        self.t = nn.ParameterList([nn.Parameter(torch.empty(h_K)) for _ in range(K)])

        # Auxiliary network for unconstrained inputs
        non_monotonic_dim = n - len(monotonic_indices)
        self.auxiliary_net = nn.Sequential(
            nn.Linear(non_monotonic_dim, aux_hidden_units),
            aux_activation,
            nn.Linear(aux_hidden_units, 1)
        )

        self.init_weights(init_method)

    def init_weights(self, method: Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal']) -> None:
        """Initialize network parameters."""
        init_weights(self, method)
        init_weights(self.auxiliary_net, method)

    def soft_max(self, a: torch.Tensor) -> torch.Tensor:
        """Compute the soft maximum."""
        return torch.logsumexp(torch.exp(self.beta) * a, dim=1) / torch.exp(self.beta)

    def soft_min(self, a: torch.Tensor) -> torch.Tensor:
        """Compute the soft minimum."""
        return -torch.logsumexp(-torch.exp(self.beta) * a, dim=1) / torch.exp(self.beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size,).
        """
        x_unconstrained = x[:, ~self.monotonic_mask]
        aux_output = self.auxiliary_net(x_unconstrained)

        group_outputs = []
        for i in range(self.K):
            w = torch.where(self.monotonic_mask, transform_weights(self.z[i], self.transform), self.z[i])
            a = torch.matmul(x, w.t()) + self.t[i] + aux_output.squeeze(-1)
            g = self.soft_max(a)
            group_outputs.append(g)

        y = self.soft_min(torch.stack(group_outputs, dim=1))
        return torch.sigmoid(y) if self.use_sigmoid else y

    def check_grad(self) -> int:
        """
        Count the number of parameters with zero gradients.

        Returns:
            int: Number of parameters with zero gradients.
        """
        return check_grad(self) + check_grad(self.auxiliary_net)

    def check_grad_neuron(self) -> int:
        """
        Count the number of neurons with all zero gradients.

        Returns:
            int: Number of neurons with all zero gradients.
        """
        return sum(check_grad_neuron(self.z[i], self.t[i]) for i in range(self.K))

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
        init_method: Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal'] = 'xavier_uniform',
        transform: Literal['exp', 'explin', 'sqr'] = 'exp',
        use_sigmoid: bool = False
    ):
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

        self.init_weights(init_method)

    def init_weights(self, method: Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal']) -> None:
        """
        Initialize network parameters.

        Args:
            method (str): Weight initialization method.
        """
        self._init_weights(self, method)

    @staticmethod
    def _init_weights(module: nn.Module, method: Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal']) -> None:
        """
        Static method to initialize weights of a module.

        Args:
            module (nn.Module): The module whose weights to initialize.
            method (str): Weight initialization method.
        """
        if method.startswith('xavier'):
            gain = nn.init.calculate_gain('tanh')
            init_func = nn.init.xavier_uniform_ if method.endswith('uniform') else nn.init.xavier_normal_
        elif method.startswith('kaiming'):
            init_func = nn.init.kaiming_uniform_ if method.endswith('uniform') else nn.init.kaiming_normal_
            gain = 1.0
        else:
            raise ValueError(f"Unsupported initialization method: {method}")

        for param in module.parameters():
            if len(param.shape) > 1:  # weights
                init_func(param, gain=gain)
            else:  # biases
                nn.init.zeros_(param)

    @staticmethod
    def transform_weights(weights: torch.Tensor, method: Literal['exp', 'explin', 'sqr']) -> torch.Tensor:
        """
        Apply the specified transformation to ensure positive weights.

        Args:
            weights (torch.Tensor): Input weights.
            method (str): Transformation method.

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network. To be implemented by subclasses.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size,).
        """
        raise NotImplementedError("Subclasses must implement forward method")

    def check_grad(self) -> int:
        """
        Count the number of parameters with zero gradients.

        Returns:
            int: Number of parameters with zero gradients.
        """
        return sum(torch.sum((param.grad == 0).int()).item() for param in self.parameters() if param.grad is not None)

    def check_grad_neuron(self) -> int:
        """
        Count the number of neurons with all zero gradients.

        Returns:
            int: Number of neurons with all zero gradients.
        """
        return sum(self._check_grad_neuron(self.z[i], self.t[i]) for i in range(self.K))

    @staticmethod
    def _check_grad_neuron(weights: torch.Tensor, biases: torch.Tensor) -> int:
        """
        Static method to count the number of neurons with all zero gradients.

        Args:
            weights (torch.Tensor): Weight tensor.
            biases (torch.Tensor): Bias tensor.

        Returns:
            int: Number of neurons with all zero gradients.
        """
        weights_zero = torch.all(weights.grad == 0, dim=1).int()
        biases_zero = (biases.grad == 0).int()
        return torch.sum(weights_zero * biases_zero).item()

class MinMaxNetwork(MinMaxNetworkBase):
    """
    MinMaxNetwork implementation with mask for non-monotonic features.
    """

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
            w = torch.where(self.monotonic_mask, self.transform_weights(self.z[i], self.transform), self.z[i])
            a = torch.matmul(x, w.t()) + self.t[i]
            g, _ = torch.max(a, dim=1)
            group_outputs.append(g)

        y = torch.min(torch.stack(group_outputs), dim=0)[0]
        return torch.sigmoid(y) if self.use_sigmoid else y

class SmoothMinMaxNetwork(MinMaxNetworkBase):
    """
    SmoothMinMaxNetwork implementation with mask for non-monotonic features.
    """

    def __init__(self, *args, beta: float = -1.0, **kwargs):
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
            w = torch.where(self.monotonic_mask, self.transform_weights(self.z[i], self.transform), self.z[i])
            a = torch.matmul(x, w.t()) + self.t[i]
            g = self.soft_max(a)
            group_outputs.append(g)

        y = self.soft_min(torch.stack(group_outputs, dim=1))
        return torch.sigmoid(y) if self.use_sigmoid else y

class MinMaxNetworkWithMLP(MinMaxNetworkBase):
    """
    MinMaxNetwork with auxiliary MLP for partially monotone problems.
    """

    def __init__(self, *args, aux_hidden_units: int = 64, aux_activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU(), **kwargs):
        """
        Initialize the MinMaxNetworkWithMLP.

        Args:
            *args: Variable length argument list.
            aux_hidden_units (int): Number of hidden units in the auxiliary network.
            aux_activation (Callable[[torch.Tensor], torch.Tensor]): Activation function for the auxiliary network.
            **kwargs: Arbitrary keyword arguments.
        """
        super(MinMaxNetworkWithMLP, self).__init__(*args, **kwargs)
        non_monotonic_dim = sum(~self.monotonic_mask)
        self.auxiliary_net = nn.Sequential(
            nn.Linear(non_monotonic_dim, aux_hidden_units),
            aux_activation,
            nn.Linear(aux_hidden_units, 1)
        )
        self._init_weights(self.auxiliary_net, kwargs.get('init_method', 'xavier_uniform'))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size,).
        """
        x_unconstrained = x[:, ~self.monotonic_mask]
        aux_output = self.auxiliary_net(x_unconstrained)

        group_outputs = []
        for i in range(self.K):
            w = torch.where(self.monotonic_mask, self.transform_weights(self.z[i], self.transform), self.z[i])
            a = torch.matmul(x, w.t()) + self.t[i] + aux_output.squeeze(-1)
            g, _ = torch.max(a, dim=1)
            group_outputs.append(g)

        y = torch.min(torch.stack(group_outputs), dim=0)[0]
        return torch.sigmoid(y) if self.use_sigmoid else y

    def check_grad(self) -> int:
        """
        Count the number of parameters with zero gradients, including the auxiliary network.

        Returns:
            int: Number of parameters with zero gradients.
        """
        return super().check_grad() + sum(torch.sum((param.grad == 0).int()).item() for param in self.auxiliary_net.parameters() if param.grad is not None)

class SmoothMinMaxNetworkWithMLP(SmoothMinMaxNetwork, MinMaxNetworkWithMLP):
    """
    SmoothMinMaxNetwork with auxiliary MLP for partially monotone problems.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the SmoothMinMaxNetworkWithMLP.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        SmoothMinMaxNetwork.__init__(self, *args, **kwargs)
        MinMaxNetworkWithMLP.__init__(self, *args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size,).
        """
        x_unconstrained = x[:, ~self.monotonic_mask]
        aux_output = self.auxiliary_net(x_unconstrained)

        group_outputs = []
        for i in range(self.K):
            w = torch.where(self.monotonic_mask, self.transform_weights(self.z[i], self.transform), self.z[i])
            a = torch.matmul(x, w.t()) + self.t[i] + aux_output.squeeze(-1)
            g = self.soft_max(a)
            group_outputs.append(g)

        y = self.soft_min(torch.stack(group_outputs, dim=1))
        return torch.sigmoid(y) if self.use_sigmoid else y

    def check_grad(self) -> int:
        """
        Count the number of parameters with zero gradients, including the auxiliary network.

        Returns:
            int: Number of parameters with zero gradients.
        """
        return SmoothMinMaxNetwork.check_grad(self) + sum(torch.sum((param.grad == 0).int()).item() for param in self.auxiliary_net.parameters() if param.grad is not None)