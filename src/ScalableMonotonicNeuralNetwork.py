import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
from src.utils import init_weights

class ScalableMonotonicNeuralNetwork(nn.Module):
    class ActivationLayer(nn.Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.weight = nn.Parameter(torch.empty((in_features, out_features)))
            self.bias = nn.Parameter(torch.empty(out_features))

        def forward(self, x):
            raise NotImplementedError("Abstract method called")

    class ExpUnit(ActivationLayer):
        def __init__(self, in_features: int, out_features: int):
            super().__init__(in_features, out_features)
            nn.init.uniform_(self.weight, a=-20.0, b=2.0)
            init_weights(self.bias, method='truncated_normal', std=0.5)

        def forward(self, x):
            out = x @ torch.exp(self.weight) + self.bias
            return (1 - 0.01) * torch.clip(out, 0, 1) + 0.01 * out

    class ReLUUnit(ActivationLayer):
        def __init__(self, in_features: int, out_features: int):
            super().__init__(in_features, out_features)
            init_weights(self, method='xavier_uniform')
            init_weights(self.bias, method='truncated_normal', std=0.5)

        def forward(self, x):
            return F.relu(x @ self.weight + self.bias)

    class ConfluenceUnit(ActivationLayer):
        def __init__(self, in_features: int, out_features: int):
            super().__init__(in_features, out_features)
            init_weights(self, method='xavier_uniform')
            init_weights(self.bias, method='truncated_normal', std=0.5)

        def forward(self, x):
            out = x @ self.weight + self.bias
            return (1 - 0.01) * torch.clip(out, 0, 1) + 0.01 * out

    class FCLayer(ActivationLayer):
        def __init__(self, in_features: int, out_features: int):
            super().__init__(in_features, out_features)
            init_weights(self.weight, method='truncated_normal', mean=-10.0, std=3)
            init_weights(self.bias, method='truncated_normal', std=0.5)

        def forward(self, x):
            return x @ torch.exp(self.weight) + self.bias

    def __init__(self,
                 input_size: int,
                 mono_feature: List[int],
                 exp_unit_size: Tuple = (),
                 relu_unit_size: Tuple = (),
                 conf_unit_size: Tuple = ()):
        super(ScalableMonotonicNeuralNetwork, self).__init__()

        self.input_size = input_size
        self.mono_feature = mono_feature
        self.mono_size = len(mono_feature)
        self.non_mono_size = input_size - self.mono_size
        self.non_mono_feature = list(set(range(input_size)) - set(mono_feature))
        self.exp_unit_size = exp_unit_size
        self.relu_unit_size = relu_unit_size
        self.conf_unit_size = conf_unit_size

        self.exp_units = nn.ModuleList([
            self.ExpUnit(self.mono_size if i == 0 else exp_unit_size[i - 1] + conf_unit_size[i - 1], exp_unit_size[i])
            for i in range(len(exp_unit_size))
        ])

        self.relu_units = nn.ModuleList([
            self.ReLUUnit(self.non_mono_size if i == 0 else relu_unit_size[i - 1], relu_unit_size[i])
            for i in range(len(relu_unit_size))
        ])

        self.conf_units = nn.ModuleList([
            self.ConfluenceUnit(self.non_mono_size if i == 0 else relu_unit_size[i - 1], conf_unit_size[i])
            for i in range(len(relu_unit_size))
        ])

        self.fc_layer = self.FCLayer(
            exp_unit_size[-1] + conf_unit_size[-1] + relu_unit_size[-1], 1
        )

    def forward(self, x):
        x_mono = x[:, self.mono_feature]
        x_non_mono = x[:, self.non_mono_feature]

        exp_output = x_mono
        relu_output = x_non_mono

        for i in range(len(self.exp_unit_size)):
            conf_output = self.conf_units[i](relu_output)
            exp_output = self.exp_units[i](torch.cat([exp_output, conf_output], dim=1))
            relu_output = self.relu_units[i](relu_output)

        out = self.fc_layer(torch.cat([conf_output, exp_output, relu_output], dim=1))
        return out

    def count_parameters(self) -> int:
        """
        Count the number of trainable parameters in the network.

        Returns:
            int: Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)