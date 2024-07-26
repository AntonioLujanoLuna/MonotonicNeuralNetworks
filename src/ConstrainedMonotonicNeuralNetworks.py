import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Callable, Literal, List
from contextlib import contextmanager
from functools import lru_cache

from src.MLP import StandardMLP
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


class ConstrainedMonotonicNeuralNetwork(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_sizes: List[int],
                 output_size: int,
                 activation: str = 'elu',
                 monotonicity_indicator: List[int] = None,
                 final_activation: Optional[Callable] = None,
                 dropout_rate: float = 0.0,
                 init_method: Literal[
                     'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'he_uniform', 'he_normal', 'truncated_normal'] = 'xavier_uniform',
                 architecture_type: Literal['type1', 'type2'] = 'type1'):
        super(ConstrainedMonotonicNeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation
        self.monotonicity_indicator = monotonicity_indicator or [1] * input_size
        self.final_activation = final_activation
        self.dropout_rate = dropout_rate
        self.architecture_type = architecture_type

        if len(self.monotonicity_indicator) != input_size:
            raise ValueError(
                f"Length of monotonicity_indicator ({len(self.monotonicity_indicator)}) must match input_size ({input_size})")

        if architecture_type == 'type1':
            self.network = self._build_type1()
        elif architecture_type == 'type2':
            self.network = self._build_type2()
        else:
            raise ValueError("architecture_type must be either 'type1' or 'type2'")

        self.init_weights(init_method)

    def _build_type1(self):
        layers = nn.ModuleList()

        # Input layer
        layers.append(MonoDense(
            in_features=self.input_size,
            units=self.hidden_sizes[0],
            activation=self.activation,
            monotonicity_indicator=self.monotonicity_indicator
        ))

        # Hidden layers
        for i in range(1, len(self.hidden_sizes)):
            layers.append(MonoDense(
                in_features=self.hidden_sizes[i - 1],
                units=self.hidden_sizes[i],
                activation=self.activation,
                monotonicity_indicator=1
            ))

        # Output layer
        layers.append(MonoDense(
            in_features=self.hidden_sizes[-1],
            units=self.output_size,
            activation=None,
            monotonicity_indicator=1
        ))

        return layers

    def _build_type2(self):
        mono_layers = nn.ModuleList()
        non_mono_features = []

        for i, indicator in enumerate(self.monotonicity_indicator):
            if indicator != 0:
                mono_layers.append(MonoDense(
                    in_features=1,
                    units=self.hidden_sizes[0],
                    activation=self.activation,
                    monotonicity_indicator=indicator
                ))
            else:
                non_mono_features.append(i)

        non_mono_input_size = len(non_mono_features)
        if non_mono_input_size > 0:
            non_mono_mlp = StandardMLP(
                input_size=non_mono_input_size,
                hidden_sizes=[self.hidden_sizes[0]],
                output_size=self.hidden_sizes[0],
                activation=self._get_activation(self.activation),
                dropout_rate=self.dropout_rate
            )
        else:
            non_mono_mlp = None

        main_input_size = self.hidden_sizes[0] * (len(mono_layers) + (1 if non_mono_mlp else 0))
        main_network = nn.ModuleList([
            MonoDense(
                in_features=main_input_size if i == 0 else self.hidden_sizes[i - 1],
                units=size,
                activation=self.activation if i < len(self.hidden_sizes) - 1 else None,
                monotonicity_indicator=1
            ) for i, size in enumerate(self.hidden_sizes[1:] + [self.output_size])
        ])

        return nn.ModuleDict({
            'mono_layers': mono_layers,
            'non_mono_mlp': non_mono_mlp,
            'main_network': main_network,
            'non_mono_indices': non_mono_features
        })

    def _get_activation(self, activation):
        if activation == 'elu':
            return nn.ELU()
        elif activation == 'relu':
            return nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def init_weights(self, method):
        for module in self.modules():
            if isinstance(module, (nn.Linear, MonoDense)):
                for params in module.parameters():
                    if len(params.shape) > 1:
                        init_weights(params, method=method)
                    else:
                        init_weights(params, method='zeros')

    def forward(self, x: torch.Tensor):
        if self.architecture_type == 'type1':
            for layer in self.network:
                x = layer(x)
        elif self.architecture_type == 'type2':
            mono_outputs = [layer(x[:, i].unsqueeze(1)) for i, layer in enumerate(self.network['mono_layers'])]

            if self.network['non_mono_mlp']:
                non_mono_input = x[:, self.network['non_mono_indices']]
                non_mono_output = self.network['non_mono_mlp'](non_mono_input)
                all_outputs = mono_outputs + [non_mono_output]
            else:
                all_outputs = mono_outputs

            x = torch.cat(all_outputs, dim=-1)

            for layer in self.network['main_network']:
                x = layer(x)

        if self.final_activation:
            x = self.final_activation(x)

        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
