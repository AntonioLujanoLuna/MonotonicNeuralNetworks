import torch
import torch.nn as nn
from typing import Callable, List, Literal
from src.utils import init_weights, transform_weights


class MinMaxNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        K: int,
        h_K: int,
        monotonic_indices: List[int],
        device: torch.device,
        init_method: Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal',
            'he_uniform', 'he_normal', 'truncated_normal'] = 'xavier_uniform',
        transform: Literal['exp', 'explin', 'sqr'] = 'exp',
        use_sigmoid: bool = False,
    ):
        """
        MinMaxNetwork implementation with mask for non-monotonic features.

        Args:
            input_size (int): Number of inputs.
            K (int): Number of groups.
            h_K (int): Number of neurons per group.
            monotonic_indices (List[int]): Indices of monotonic features.
            device (torch.device): Device
            init_method (Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'he_uniform', 'he_normal', 'truncated_normal']): Weight initialization method.
            transform (Literal['exp', 'explin', 'sqr']): Type of transformation for ensuring positivity.
            use_sigmoid (bool): Whether to apply sigmoid to the output.
        """
        super(MinMaxNetwork, self).__init__()
        self.input_size = input_size
        self.K = K
        self.h_K = h_K
        self.device = device
        self.monotonic_mask = torch.zeros(input_size, dtype=torch.bool, device=self.device)
        self.monotonic_mask[monotonic_indices] = True
        self.transform = transform
        self.use_sigmoid = use_sigmoid
        self.z = nn.ParameterList([nn.Parameter(torch.empty(h_K, input_size, device=self.device)) for _ in range(K)])
        self.t = nn.ParameterList([nn.Parameter(torch.empty(h_K, device=self.device)) for _ in range(K)])

        for params in self.parameters():
            if len(params.shape) > 1:
                init_weights(params, method=init_method)
            else:
                init_weights(params, method='zeros')

    def init_weights(self, method: Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'he_uniform', 'he_normal', 'truncated_normal']) -> None:
        """Initialize network parameters."""
        for params in self.parameters():
            if len(params.shape) > 1:
                init_weights(params, method=method)
            else:
                init_weights(params, method='zeros')

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
        y = y.view(-1, 1)  # Reshape to (batch_size, 1)
        return torch.sigmoid(y) if self.use_sigmoid else y

class MinMaxNetworkWithMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        K: int,
        h_K: int,
        monotonic_indices: List[int],
        device: torch.device,
        aux_hidden_units: int = 64,
        init_method: Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'he_uniform', 'he_normal', 'truncated_normal'] = 'xavier_uniform',
        transform: Literal['exp', 'explin', 'sqr'] = 'exp',
        use_sigmoid: bool = False,
        aux_activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()
    ):
        """
        MinMaxNetwork with auxiliary MLP for partially monotone problems.

        Args:
            input_size (int): Number of inputs.
            K (int): Number of groups.
            h_K (int): Number of neurons per group.
            monotonic_indices (List[int]): Indices of monotonic features.
            device (torch.device): Device
            aux_hidden_units (int): Number of hidden units in the auxiliary network.
            init_method (Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'he_uniform', 'he_normal', 'truncated_normal']): Weight initialization method.
            transform (Literal['exp', 'explin', 'sqr']): Type of transformation for ensuring positivity.
            use_sigmoid (bool): Whether to apply sigmoid to the output.
            aux_activation (Callable[[torch.Tensor], torch.Tensor]): Activation function for the auxiliary network.
        """
        super(MinMaxNetworkWithMLP, self).__init__()
        self.input_size = input_size
        self.K = K
        self.h_K = h_K
        self.device = device
        self.monotonic_mask = torch.zeros(input_size, dtype=torch.bool)
        self.monotonic_mask[monotonic_indices] = True
        self.transform = transform
        self.use_sigmoid = use_sigmoid

        self.z = nn.ParameterList([nn.Parameter(torch.empty(h_K, input_size, device=self.device)) for _ in range(K)])
        self.t = nn.ParameterList([nn.Parameter(torch.empty(h_K, device=self.device)) for _ in range(K)])

        # Auxiliary network for unconstrained inputs
        non_monotonic_dim = input_size - len(monotonic_indices)
        self.auxiliary_net = nn.Sequential(
            nn.Linear(non_monotonic_dim, aux_hidden_units),
            aux_activation,
            nn.Linear(aux_hidden_units, 1)
        )

        for params in self.parameters():
            if len(params.shape) > 1:
                init_weights(params, method=init_method)
            else:
                init_weights(params, method='zeros')

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
        y = y.view(-1, 1)  # Reshape to (batch_size, 1)
        return torch.sigmoid(y) if self.use_sigmoid else y


class SmoothMinMaxNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        K: int,
        h_K: int,
        monotonic_indices: List[int],
        device: torch.device,
        beta: float = -1.0,
        init_method: Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'he_uniform', 'he_normal', 'truncated_normal'] = 'xavier_uniform',
        transform: Literal['exp', 'explin', 'sqr'] = 'exp',
        use_sigmoid: bool = False
    ):
        """
        SmoothMinMaxNetwork implementation with mask for non-monotonic features.

        Args:
            input_size (int): Number of inputs.
            K (int): Number of groups.
            h_K (int): Number of neurons per group.
            monotonic_indices (List[int]): Indices of monotonic features.
            device (torch.device): Device
            beta (float): Initial value for the smoothing parameter.
            init_method (Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'he_uniform', 'he_normal', 'truncated_normal']): Weight initialization method.
            transform (Literal['exp', 'explin', 'sqr']): Type of transformation for ensuring positivity.
            use_sigmoid (bool): Whether to apply sigmoid to the output.
        """
        super(SmoothMinMaxNetwork, self).__init__()
        self.input_size = input_size
        self.K = K
        self.h_K = h_K
        self.device = device
        self.monotonic_mask = torch.zeros(input_size, dtype=torch.bool)
        self.monotonic_mask[monotonic_indices] = True
        self.transform = transform
        self.use_sigmoid = use_sigmoid

        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float))
        self.z = nn.ParameterList([nn.Parameter(torch.empty(h_K, input_size, device=self.device)) for _ in range(K)])
        self.t = nn.ParameterList([nn.Parameter(torch.empty(h_K, device=self.device)) for _ in range(K)])

        for params in self.parameters():
            if len(params.shape) > 1:
                init_weights(params, method=init_method)
            else:
                init_weights(params, method='zeros')


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
        y = y.view(-1, 1)  # Reshape to (batch_size, 1)
        return torch.sigmoid(y) if self.use_sigmoid else y

class SmoothMinMaxNetworkWithMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        K: int,
        h_K: int,
        monotonic_indices: List[int],
        device: torch.device,
        aux_hidden_units: int = 64,
        beta: float = -1.0,
        init_method: Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'he_uniform', 'he_normal', 'truncated_normal'] = 'xavier_uniform',
        transform: Literal['exp', 'explin', 'sqr'] = 'exp',
        use_sigmoid: bool = False,
        aux_activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()
    ):
        """
        SmoothMinMaxNetwork with auxiliary MLP for partially monotone problems.

        Args:
            input_size (int): Number of inputs.
            K (int): Number of groups.
            h_K (int): Number of neurons per group.
            monotonic_indices (List[int]): Indices of monotonic features.
            device (torch.device): Device
            aux_hidden_units (int): Number of hidden units in the auxiliary network.
            beta (float): Initial value for the smoothing parameter.
            init_method (Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'he_uniform', 'he_normal', 'truncated_normal']): Weight initialization method.
            transform (Literal['exp', 'explin', 'sqr']): Type of transformation for ensuring positivity.
            use_sigmoid (bool): Whether to apply sigmoid to the output.
            aux_activation (Callable[[torch.Tensor], torch.Tensor]): Activation function for the auxiliary network.
        """
        super(SmoothMinMaxNetworkWithMLP, self).__init__()
        self.input_size = input_size
        self.K = K
        self.h_K = h_K
        self.device = device
        self.monotonic_mask = torch.zeros(input_size, dtype=torch.bool)
        self.monotonic_mask[monotonic_indices] = True
        self.transform = transform
        self.use_sigmoid = use_sigmoid

        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float))
        self.z = nn.ParameterList([nn.Parameter(torch.empty(h_K, input_size, device=self.device)) for _ in range(K)])
        self.t = nn.ParameterList([nn.Parameter(torch.empty(h_K, device=self.device)) for _ in range(K)])

        # Auxiliary network for unconstrained inputs
        non_monotonic_dim = input_size - len(monotonic_indices)
        self.auxiliary_net = nn.Sequential(
            nn.Linear(non_monotonic_dim, aux_hidden_units),
            aux_activation,
            nn.Linear(aux_hidden_units, 1)
        )

        for params in self.parameters():
            if len(params.shape) > 1:
                init_weights(params, method=init_method)
            else:
                init_weights(params, method='zeros')


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
        y = y.view(-1, 1)  # Reshape to (batch_size, 1)
        return torch.sigmoid(y) if self.use_sigmoid else y
