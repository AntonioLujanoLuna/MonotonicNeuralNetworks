
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Callable, Literal, List
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
        if is_convex and is_concave:
            raise ValueError("The model cannot be set to be both convex and concave (only linear functions are both).")
        super(MonoDense, self).__init__()
        self.in_features = in_features
        self.init_method = init_method
        self.units = units
        self.org_activation = activation
        self.is_convex = is_convex
        self.is_concave = is_concave
        self.activation_weights = nn.Parameter(torch.tensor(activation_weights))
        self.monotonicity_indicator = self.get_monotonicity_indicator(
            monotonicity_indicator, self.in_features, self.units
        )
        self.weight = nn.Parameter(torch.Tensor(units, in_features))
        self.bias = nn.Parameter(torch.Tensor(units))
        self.reset_parameters()
        self.convex_activation, self.concave_activation, self.saturated_activation = self.get_activation_functions(
            self.org_activation)

    def to(self, device):
        super().to(device)
        self.weight = self.weight.to(device)
        self.bias = self.bias.to(device)
        self.activation_weights = self.activation_weights.to(device)
        self.monotonicity_indicator = self.monotonicity_indicator.to(device)
        return self

    def reset_parameters(self):
        with torch.no_grad():
            # Create new tensors for weight and bias
            new_weight = torch.empty(self.weight.shape, device=self.weight.device)
            new_bias = torch.empty(self.bias.shape, device=self.bias.device)

            # Initialize the new tensors
            init_weights(new_weight, method=self.init_method)
            init_weights(new_bias, method='zeros')

            # Create new nn.Parameter objects and assign them
            self.weight = nn.Parameter(new_weight)
            self.bias = nn.Parameter(new_bias)


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
        if isinstance(monotonicity_indicator, torch.Tensor):
            monotonicity_indicator = monotonicity_indicator.clone().detach().to(torch.float32)
        else:
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
            # Add a small epsilon to avoid division by zero
            epsilon = 1e-8
            activation_weights_sum = self.activation_weights.sum() + epsilon
            normalized_activation_weights = self.activation_weights / activation_weights_sum

        # Ensure normalized_activation_weights are not NaN
        normalized_activation_weights = torch.nan_to_num(normalized_activation_weights, nan=1.0 / 3)

        s_convex = max(0, min(self.units, round(normalized_activation_weights[0].item() * self.units)))
        s_concave = max(0, min(self.units - s_convex, round(normalized_activation_weights[1].item() * self.units)))
        s_saturated = self.units - s_convex - s_concave

        h_convex, h_concave, h_saturated = torch.split(h, [s_convex, s_concave, s_saturated], dim=-1)

        y_convex = self.convex_activation(h_convex) if s_convex > 0 else torch.tensor([], device=h.device)
        y_concave = self.concave_activation(h_concave) if s_concave > 0 else torch.tensor([], device=h.device)
        y_saturated = self.saturated_activation(h_saturated) if s_saturated > 0 else torch.tensor([], device=h.device)

        y = torch.cat([y_convex, y_concave, y_saturated], dim=-1)

        return y
    def forward(self, x):
        device = x.device
        self.monotonicity_indicator = self.monotonicity_indicator.to(device)
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
                 device: torch.device,
                 activation: str = 'elu',
                 monotonicity_indicator: List[int] = None,
                 final_activation: Optional[Callable] = None,
                 init_method: Literal[
                     'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'he_uniform', 'he_normal', 'truncated_normal'] = 'xavier_uniform',
                 architecture_type: Literal['type1', 'type2'] = 'type1'):
        super(ConstrainedMonotonicNeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation
        self.final_activation = final_activation
        self.device = device
        self.monotonicity_indicator = nn.Parameter(torch.tensor(monotonicity_indicator, dtype=torch.float32), requires_grad=False)
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

    def to(self, device):
        super().to(device)
        self.monotonicity_indicator = self.monotonicity_indicator.to(device)
        if self.architecture_type == 'type1':
            self.network = self.network.to(device)
        elif self.architecture_type == 'type2':
            self.network['mono_layers'] = self.network['mono_layers'].to(device)
            self.network['non_mono_layers'] = self.network['non_mono_layers'].to(device)
            self.network['main_network'] = self.network['main_network'].to(device)
        return self

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
        non_mono_layers = nn.ModuleList()

        for i, indicator in enumerate(self.monotonicity_indicator):
            if indicator != 0:
                mono_layers.append(MonoDense(
                    in_features=1,
                    units=self.hidden_sizes[0],
                    activation=self.activation,
                    monotonicity_indicator=torch.tensor([indicator])
                ))
            else:
                non_mono_layers.append(nn.Sequential(
                    nn.Linear(1, self.hidden_sizes[0]),
                    self._get_activation_layer(self.activation)
                ))

        main_input_size = self.hidden_sizes[0] * self.input_size

        main_network = nn.ModuleList()

        for i in range(1, len(self.hidden_sizes)):
            main_network.append(MonoDense(
                in_features=self.hidden_sizes[i - 1] if i != 1 else main_input_size,
                units=self.hidden_sizes[i],
                activation=self.activation,
                monotonicity_indicator=1
            ))

        main_network.append(MonoDense(
            in_features=self.hidden_sizes[-1],
            units=self.output_size,
            activation=None,
            monotonicity_indicator=1
        ))

        return nn.ModuleDict({
            'mono_layers': mono_layers,
            'non_mono_layers': non_mono_layers,
            'main_network': main_network
        })

    def _get_activation_layer(self, activation):
        if activation == 'elu':
            return nn.ELU()
        elif activation == 'relu':
            return nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def init_weights(self, method):
        for module in self.modules():
            if isinstance(module, (nn.Linear, MonoDense)):
                for name, param in module.named_parameters():
                    if param.dim() > 1:
                        # Create a new tensor and initialize it
                        new_param = torch.empty_like(param)
                        init_weights(new_param, method=method)
                        # Assign the new tensor to the parameter
                        module._parameters[name] = nn.Parameter(new_param)
                    else:
                        # For biases, we can directly initialize them to zeros
                        with torch.no_grad():
                            param.zero_()

    def forward(self, x: torch.Tensor):
        if self.architecture_type == 'type2':
            # Identify monotonic and non-monotonic inputs
            monotonic_mask = (self.monotonicity_indicator != 0)
            monotonic_inputs = x[:, monotonic_mask]
            non_monotonic_inputs = x[:, ~monotonic_mask]
            # Process monotonic inputs
            if monotonic_inputs.size(1) > 0:
                # Stack the inputs along a new dimension to process in parallel
                mono_inputs_expanded = monotonic_inputs.unsqueeze(2)  # Shape: (batch_size, num_features, 1)
                mono_outputs = torch.stack([
                    layer(mono_inputs_expanded[:, i, :])
                    for i, layer in enumerate(self.network['mono_layers'])
                ], dim=2)  # Shape: (batch_size, hidden_size, num_features)
                mono_outputs = mono_outputs.view(x.size(0), -1)  # Flatten to (batch_size, hidden_size * num_features)
            else:
                mono_outputs = torch.tensor([], device=x.device)
            # Process non-monotonic inputs similarly
            if non_monotonic_inputs.size(1) > 0:
                non_mono_inputs_expanded = non_monotonic_inputs.unsqueeze(2)
                non_mono_outputs = torch.stack([
                    layer(non_mono_inputs_expanded[:, i, :])
                    for i, layer in enumerate(self.network['non_mono_layers'])
                ], dim=2)
                non_mono_outputs = non_mono_outputs.view(x.size(0), -1)
            else:
                non_mono_outputs = torch.tensor([], device=x.device)
            # Concatenate the outputs of both monotonic and non-monotonic layers
            x = torch.cat((mono_outputs, non_mono_outputs), dim=1)
            # Process through the main network layers
            for layer in self.network['main_network']:
                x = layer(x)

        if self.final_activation:
            x = self.final_activation(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
