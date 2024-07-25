import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Callable, Literal
from contextlib import contextmanager
from functools import lru_cache
from src.utils import init_weights

class MonoDense(nn.Module):
    def __init__(
            self,
            in_features: int,
            units: int,
            activation: Optional[Union[str, Callable]] = None,
            monotonicity_indicator: Union[int, list] = 1,
            is_convex: bool = False,
            is_concave: bool = False,
            activation_weights: Tuple[float, float, float] = (7.0, 7.0, 2.0),
            init_method: Literal[
                'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'he_uniform', 'he_normal', 'truncated_normal'] = 'xavier_uniform'
    ):
        super(MonoDense, self).__init__()
        self.in_features = in_features
        self.init_method = init_method
        self.units = units
        self.org_activation = activation
        self.is_convex = is_convex
        self.is_concave = is_concave
        self.activation_weights = torch.tensor(activation_weights)
        self.monotonicity_indicator = monotonicity_indicator

        self.weight = nn.Parameter(torch.Tensor(units, in_features))
        self.bias = nn.Parameter(torch.Tensor(units))

        self.built = False
        self.build()

    def build(self):
        if not self.built:
            self.monotonicity_indicator = self.get_monotonicity_indicator(
                self.monotonicity_indicator, self.in_features, self.units
            )
            self.reset_parameters()
            self.convex_activation, self.concave_activation, self.saturated_activation = self.get_activation_functions(
                self.org_activation)
            self.built = True

    def reset_parameters(self):
        for params in self.parameters():
            if len(params.shape) > 1:
                init_weights(params, method=self.init_method)
            else:
                init_weights(params, method='zeros')


    def get_config(self):
        return {
            'in_features': self.in_features,
            'units': self.units,
            'init_method': self.init_method,
            'activation': self.org_activation,
            'monotonicity_indicator': self.monotonicity_indicator.tolist(),
            'is_convex': self.is_convex,
            'is_concave': self.is_concave,
            'activation_weights': self.activation_weights.tolist(),
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @staticmethod
    def get_monotonicity_indicator(monotonicity_indicator, in_features, units):
        monotonicity_indicator = torch.tensor(monotonicity_indicator, dtype=torch.float32)
        if monotonicity_indicator.dim() < 2:
            monotonicity_indicator = monotonicity_indicator.reshape(-1, 1)
        elif monotonicity_indicator.dim() > 2:
            raise ValueError(f"monotonicity_indicator has rank greater than 2: {monotonicity_indicator.shape}")

        monotonicity_indicator = monotonicity_indicator.expand(in_features, units).t()

        if not torch.all(
                (monotonicity_indicator == -1) | (monotonicity_indicator == 0) | (monotonicity_indicator == 1)):
            raise ValueError(
                f"Each element of monotonicity_indicator must be one of -1, 0, 1, but it is: '{monotonicity_indicator}'")

        return monotonicity_indicator

    @staticmethod
    @lru_cache(maxsize=None)
    def get_activation_functions(activation):
        if callable(activation):
            return activation, lambda x: -activation(-x), MonoDense.get_saturated_activation(activation,
                                                                                             lambda x: -activation(-x))

        if isinstance(activation, str):
            activation = activation.lower()

        activations = {
            'relu': F.relu,
            'elu': F.elu,
            'selu': F.selu,
            'gelu': F.gelu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
            None: lambda x: x,
            'linear': lambda x: x
        }

        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")

        convex = activations[activation]
        concave_activation = lambda x: -convex(-x)
        saturated_activation = MonoDense.get_saturated_activation(convex, concave_activation)

        return convex, concave_activation, saturated_activation

    @staticmethod
    def get_saturated_activation(
            convex_activation: Callable[[torch.Tensor], torch.Tensor],
            concave_activation: Callable[[torch.Tensor], torch.Tensor],
            a: float = 1.0,
            c: float = 1.0,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        def saturated_activation(
                x: torch.Tensor,
                convex_activation: Callable[[torch.Tensor], torch.Tensor] = convex_activation,
                concave_activation: Callable[[torch.Tensor], torch.Tensor] = concave_activation,
                a: float = a,
                c: float = c,
        ) -> torch.Tensor:
            cc = convex_activation(torch.ones_like(x) * c)
            # ccc = concave_activation(-torch.ones_like(x) * c)
            return a * torch.where(
                x <= 0,
                convex_activation(x + c) - cc,
                concave_activation(x - c) + cc,
            )

        return saturated_activation

    def apply_monotonicity_indicator_to_kernel(self, kernel):
        abs_kernel = torch.abs(kernel)
        kernel = torch.where(self.monotonicity_indicator == 1, abs_kernel, kernel)
        kernel = torch.where(self.monotonicity_indicator == -1, -abs_kernel, kernel)
        return kernel

    @contextmanager
    def replace_kernel_using_monotonicity_indicator(self):
        original_kernel = self.weight.data.clone()
        self.weight.data = self.apply_monotonicity_indicator_to_kernel(self.weight)
        try:
            yield
        finally:
            self.weight.data = original_kernel

    def apply_activations(self, h):
        if self.org_activation is None:
            return h

        if self.is_convex and self.is_concave:
            raise ValueError("The model cannot be set to be both convex and concave (only linear functions are both).")

        if self.is_convex:
            normalized_activation_weights = torch.tensor([1.0, 0.0, 0.0], device=h.device)
        elif self.is_concave:
            normalized_activation_weights = torch.tensor([0.0, 1.0, 0.0], device=h.device)
        else:
            normalized_activation_weights = self.activation_weights / self.activation_weights.sum()

        s_convex = round(normalized_activation_weights[0].item() * self.units)
        s_concave = round(normalized_activation_weights[1].item() * self.units)
        s_saturated = self.units - s_convex - s_concave

        h_convex, h_concave, h_saturated = torch.split(h, [s_convex, s_concave, s_saturated], dim=-1)

        y_convex = self.convex_activation(h_convex)
        y_concave = self.concave_activation(h_concave)
        y_saturated = self.saturated_activation(h_saturated)

        y = torch.cat([y_convex, y_concave, y_saturated], dim=-1)

        return y

    def forward(self, x):
        modified_weight = self.apply_monotonicity_indicator_to_kernel(self.weight)
        h = F.linear(x, modified_weight, self.bias)

        if self.org_activation is None:
            return h

        return self.apply_activations(h)


class MonotonicSequential(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int,
                 activation: str = 'elu', monotonicity_indicator: list[int] = None,
                 final_activation: Optional[Callable] = None):
        super(MonotonicSequential, self).__init__()
        self.layers = nn.ModuleList()
        self.final_activation = final_activation

        # First layer with monotonicity indicator
        self.layers.append(MonoDense(
            in_features=input_size,
            units=hidden_sizes[0],
            activation=activation,
            monotonicity_indicator=monotonicity_indicator if monotonicity_indicator is not None else 1
        ))

        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            self.layers.append(MonoDense(
                in_features=hidden_sizes[i - 1],
                units=hidden_sizes[i],
                activation=activation,
                monotonicity_indicator=1  # Always increasing for hidden layers
            ))

        # Output layer
        self.layers.append(MonoDense(
            in_features=hidden_sizes[-1],
            units=output_size,
            activation=None,  # Linear activation
            monotonicity_indicator=0  # Default to no monotonicity enforcement for output layer
        ))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        if self.final_activation:
            x = self.final_activation(x)
        return x

