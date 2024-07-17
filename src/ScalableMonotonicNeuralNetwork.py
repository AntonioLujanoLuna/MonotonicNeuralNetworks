import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Literal

class ScalableMonotonicNeuralNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        mono_size: int,
        mono_indices: List[int],
        exp_unit_sizes: Tuple[int, ...] = (),
        relu_unit_sizes: Tuple[int, ...] = (),
        conf_unit_sizes: Tuple[int, ...] = ()
    ):
        """
        Scalable Monotonic Neural Network implementation.

        Args:
            input_size (int): Total number of input features.
            mono_size (int): Number of monotonic features.
            mono_indices (List[int]): Indices of monotonic features.
            exp_unit_sizes (Tuple[int, ...]): Sizes of exponential units.
            relu_unit_sizes (Tuple[int, ...]): Sizes of ReLU units.
            conf_unit_sizes (Tuple[int, ...]): Sizes of confluence units.
        """
        super().__init__()
        self.input_size = input_size
        self.mono_size = mono_size
        self.non_mono_size = input_size - mono_size
        self.mono_indices = mono_indices
        self.non_mono_indices = list(set(range(input_size)) - set(mono_indices))

        self.exp_units = nn.ModuleList([
            self.ExpUnit(mono_size if i == 0 else exp_unit_sizes[i-1] + conf_unit_sizes[i-1], exp_unit_sizes[i])
            for i in range(len(exp_unit_sizes))
        ])

        self.relu_units = nn.ModuleList([
            self.ReLUUnit(self.non_mono_size if i == 0 else relu_unit_sizes[i-1], relu_unit_sizes[i])
            for i in range(len(relu_unit_sizes))
        ])

        self.conf_units = nn.ModuleList([
            self.ConfluenceUnit(self.non_mono_size if i == 0 else relu_unit_sizes[i-1], conf_unit_sizes[i])
            for i in range(len(relu_unit_sizes))
        ])

        self.fc_layer = self.FCLayer(
            exp_unit_sizes[-1] + conf_unit_sizes[-1] + relu_unit_sizes[-1], 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        x_mono = x[:, self.mono_indices]
        x_non_mono = x[:, self.non_mono_indices]

        exp_output = x_mono
        relu_output = x_non_mono

        for i in range(len(self.exp_units)):
            conf_output = self.conf_units[i](relu_output)
            exp_output = self.exp_units[i](torch.cat([exp_output, conf_output], dim=1))
            relu_output = self.relu_units[i](relu_output)

        out = self.fc_layer(torch.cat([exp_output, relu_output], dim=1))
        return out

    @staticmethod
    def init_weights(module: nn.Module, method: Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'truncated_normal']) -> None:
        """
        Initialize weights of a module using the specified method.

        Args:
            module (nn.Module): The module whose weights to initialize.
            method (Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'truncated_normal']): Initialization method.
        """
        if method.startswith('xavier'):
            init_func = nn.init.xavier_uniform_ if method.endswith('uniform') else nn.init.xavier_normal_
        elif method.startswith('kaiming'):
            init_func = nn.init.kaiming_uniform_ if method.endswith('uniform') else nn.init.kaiming_normal_
        elif method == 'truncated_normal':
            def truncated_normal_(tensor, mean=0., std=1.):
                with torch.no_grad():
                    tensor.normal_(mean, std)
                    while True:
                        cond = (tensor < -2 * std) | (tensor > 2 * std)
                        if not torch.sum(cond):
                            break
                        tensor[cond] = tensor[cond].normal_(mean, std)
            init_func = truncated_normal_
        else:
            raise ValueError(f"Unsupported initialization method: {method}")

        for param in module.parameters():
            if len(param.shape) > 1:  # weights
                init_func(param)
            else:  # biases
                nn.init.zeros_(param)

    class ActivationLayer(nn.Module):
        def __init__(self, in_features: int, out_features: int, init_method: str):
            super().__init__()
            self.weight = nn.Parameter(torch.empty((in_features, out_features)))
            self.bias = nn.Parameter(torch.empty(out_features))
            ScalableMonotonicNeuralNetwork.init_weights(self, init_method)

        def forward(self, x):
            raise NotImplementedError("Abstract method called")

    class ExpUnit(ActivationLayer):
        def __init__(self, in_features: int, out_features: int):
            super().__init__(in_features, out_features, 'truncated_normal')
            with torch.no_grad():
                self.weight.uniform_(-20.0, 2.0)
                self.bias.normal_(std=0.5)

        def forward(self, x):
            out = x @ torch.exp(self.weight) + self.bias
            return (1 - 0.01) * torch.clip(out, 0, 1) + 0.01 * out

    class ReLUUnit(ActivationLayer):
        def __init__(self, in_features: int, out_features: int):
            super().__init__(in_features, out_features, 'xavier_uniform')

        def forward(self, x):
            return F.relu(x @ self.weight + self.bias)

    class ConfluenceUnit(ActivationLayer):
        def __init__(self, in_features: int, out_features: int):
            super().__init__(in_features, out_features, 'xavier_uniform')

        def forward(self, x):
            out = x @ self.weight + self.bias
            return (1 - 0.01) * torch.clip(out, 0, 1) + 0.01 * out

    class FCLayer(ActivationLayer):
        def __init__(self, in_features: int, out_features: int):
            super().__init__(in_features, out_features, 'truncated_normal')
            with torch.no_grad():
                self.weight.normal_(mean=-10.0, std=3)
                self.bias.normal_(std=0.5)

        def forward(self, x):
            return x @ torch.exp(self.weight) + self.bias

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
        def check_layer(layer):
            weights_zero = torch.all(layer.weight.grad == 0, dim=1).int()
            biases_zero = (layer.bias.grad == 0).int()
            return torch.sum(weights_zero * biases_zero).item()

        return sum(check_layer(layer) for layer in self.modules() if isinstance(layer, self.ActivationLayer))

    def count_parameters(self) -> int:
        """
        Count the number of trainable parameters in the network.

        Returns:
            int: Total number of trainable parameters.
        """
        total_params = 0

        # Count parameters in hidden layers
        for i in range(len(self.exp_units)):
            # ExpUnit
            total_params += self.exp_units[i].weight.numel() + self.exp_units[i].bias.numel()

            # ConfluenceUnit
            total_params += self.conf_units[i].weight.numel() + self.conf_units[i].bias.numel()

            # ReLUUnit
            total_params += self.relu_units[i].weight.numel() + self.relu_units[i].bias.numel()

        # Count parameters in the output layer
        total_params += self.fc_layer.weight.numel() + self.fc_layer.bias.numel()

        return total_params