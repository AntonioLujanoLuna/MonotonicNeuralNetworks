# InterpretableLayer and MonotonicLayer from https://github.com/phineasng/mononet, slightly modified

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
import math
from typing import List, Literal

from src.utils import init_weights


class InterpretableLayer(nn.Module):
    """
    An interpretable layer that applies a weight to the input.

    Attributes:
        in_features (int): Number of input features.
        weight (Parameter): Learnable weight parameter.
        softsign (nn.Softsign): Softsign activation function.
    """

    def __init__(self, in_features: int) -> None:
        """
        Initialize the InterpretableLayer.

        Args:
            in_features (int): Number of input features.
        """
        super(InterpretableLayer, self).__init__()
        self.in_features = in_features
        self.weight = Parameter(torch.Tensor(in_features))
        self.softsign = nn.Softsign()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the weight parameter."""
        init.normal_(self.weight, mean=0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the layer.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return input * self.weight

class MonotonicLayer(nn.Module):
    """
    A monotonic layer that applies a positive function to the weights.

    Attributes:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        weight (Parameter): Learnable weight parameter.
        bias (Parameter): Learnable bias parameter.
        fn (str): Name of the positive function to use.
        pos_fn (callable): The positive function to apply to weights.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, fn: str = 'exp') -> None:
        """
        Initialize the MonotonicLayer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias (bool): Whether to use a bias term.
            fn (str): Name of the positive function to use ('exp', 'square', 'abs', 'sigmoid', or 'tanh_p1').
        """
        super(MonotonicLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.fn = fn
        self.pos_fn = self._get_pos_fn(fn)
        self.reset_parameters()

    def _get_pos_fn(self, fn: str) -> callable:
        """Get the positive function based on the provided name."""
        if fn == 'exp':
            return torch.exp
        elif fn == 'square':
            return torch.square
        elif fn == 'abs':
            return torch.abs
        elif fn == 'sigmoid':
            return torch.sigmoid
        else:
            self.fn = 'tanh_p1'
            return lambda x: torch.tanh(x) + 1.

    def reset_parameters(self) -> None:
        """Initialize the weight and bias parameters."""
        n_in = self.in_features
        mean = math.log(1./n_in) if self.fn == 'exp' else 0
        init.normal_(self.weight, mean=mean)
        if self.bias is not None:
            init.uniform_(self.bias, -1./n_in, 1./n_in)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the layer.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        ret = torch.matmul(input, torch.transpose(self.pos_fn(self.weight), 0, 1))
        if self.bias is not None:
            ret = ret + self.bias
        return ret

class MonoNet(nn.Module):
    """
    MonoNet: A neural network with unconstrained, interpretable, and monotonic layers.

    Attributes:
        unconstrained_block (nn.Sequential): Block of unconstrained layers.
        interpretable (nn.Linear): Interpretable layer.
        pre_monotonic (InterpretableLayer): Pre-monotonic interpretable layer.
        monotonic_block (nn.Sequential): Block of monotonic layers.
        output (InterpretableLayer): Output interpretable layer.
        activation (nn.Module): Activation function used in the network.
    """

    def __init__(self, num_features: int, num_classes: int, hidden_sizes: List[int] = [16, 16],
                 interpretable_size: int = 8, monotonic_sizes: List[int] = [32],
                 activation: nn.Module = nn.Tanh(),
                 init_method: Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'he_uniform', 'he_normal', 'truncated_normal'] = 'xavier_uniform'
):
        """
        Initialize the MonoNet.

        Args:
            num_features (int): Number of input features.
            num_classes (int): Number of output classes.
            hidden_sizes (List[int]): Sizes of hidden layers in the unconstrained block.
            interpretable_size (int): Size of the interpretable layer.
            monotonic_sizes (List[int]): Sizes of hidden layers in the monotonic block.
            activation (nn.Module): Activation function to use in the network.
            init_method (init_method: Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'he_uniform', 'he_normal', 'truncated_normal']): Weight initialization method.
        """
        super(MonoNet, self).__init__()

        self.unconstrained_block = self._build_unconstrained_block(num_features, hidden_sizes)
        self.interpretable = nn.Linear(hidden_sizes[-1], interpretable_size)
        self.pre_monotonic = InterpretableLayer(interpretable_size)
        self.monotonic_block = self._build_monotonic_block(interpretable_size, monotonic_sizes, num_classes)
        self.output = InterpretableLayer(num_classes)
        self.activation = activation
        for params in self.parameters():
            if len(params.shape) > 1:
                init_weights(params, method=init_method)
            else:
                init_weights(params, method='zeros')

    def _build_unconstrained_block(self, input_size: int, hidden_sizes: List[int]) -> nn.Sequential:
        """Build the unconstrained block of the network."""
        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            prev_size = size
        return nn.Sequential(*layers)

    def _build_monotonic_block(self, input_size: int, hidden_sizes: List[int], output_size: int) -> nn.Sequential:
        """Build the monotonic block of the network."""
        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(MonotonicLayer(prev_size, size))
            prev_size = size
        layers.append(MonotonicLayer(prev_size, output_size))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        for layer in self.unconstrained_block:
            x = self.activation(layer(x))
        x = self.activation(self.interpretable(x))
        x = self.pre_monotonic(x)
        x = nn.functional.softsign(x)
        for layer in self.monotonic_block[:-1]:
            x = self.activation(layer(x))
        x = self.monotonic_block[-1](x)
        x = self.output(x)
        x = nn.functional.softsign(x)
        return x

    def count_parameters(self) -> int:
        """
        Count the number of trainable parameters in the network.

        Returns:
            int: The total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)