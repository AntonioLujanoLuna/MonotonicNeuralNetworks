# implementation 2
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Callable, Tuple, List
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Callable, Tuple, List
from functools import lru_cache


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
    ):
        super(MonoDense, self).__init__()
        self.in_features = in_features
        self.units = units
        self.org_activation = activation
        self.is_convex = is_convex
        self.is_concave = is_concave
        self.activation_weights = torch.tensor(activation_weights)

        self.weight = nn.Parameter(torch.Tensor(units, in_features))
        self.bias = nn.Parameter(torch.Tensor(units))
        self.reset_parameters()

        self.monotonicity_indicator = self.get_monotonicity_indicator(
            monotonicity_indicator, in_features, units
        )

        self.convex_activation, self.concave_activation, self.saturated_activation = self.get_activation_functions(
            activation)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=1)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in ** 0.5)
        nn.init.uniform_(self.bias, -bound, bound)

    def get_monotonicity_indicator(self, monotonicity_indicator, in_features, units):
        monotonicity_indicator = torch.tensor(monotonicity_indicator, dtype=torch.float32)
        if monotonicity_indicator.dim() == 0:
            monotonicity_indicator = monotonicity_indicator.expand(in_features)
        elif monotonicity_indicator.dim() == 1:
            if len(monotonicity_indicator) != in_features:
                raise ValueError(f"Length of monotonicity_indicator {len(monotonicity_indicator)} "
                                 f"must match input size {in_features}")
        else:
            raise ValueError(
                f"monotonicity_indicator must be a scalar or 1D tensor, got shape {monotonicity_indicator.shape}")

        monotonicity_indicator = monotonicity_indicator.unsqueeze(0).expand(units, -1)

        if not torch.all(
                (monotonicity_indicator == -1) | (monotonicity_indicator == 0) | (monotonicity_indicator == 1)):
            raise ValueError(
                f"Each element of monotonicity_indicator must be one of -1, 0, 1, but it is: '{monotonicity_indicator}'")

        return monotonicity_indicator

    @staticmethod
    @lru_cache(maxsize=None)
    def get_activation_functions(activation):
        if isinstance(activation, str):
            activation = activation.lower()

        if activation == 'relu':
            convex = F.relu
        elif activation == 'elu':
            convex = F.elu
        elif activation is None or activation == 'linear':
            convex = lambda x: x
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        def concave_activation(x):
            return -convex(-x)

        def saturated_activation(x, a=1.0, c=1.0):
            cc = convex(torch.ones_like(x) * c)
            ccc = concave_activation(-torch.ones_like(x) * c)
            return a * torch.where(
                x <= 0,
                convex(x + c) - cc,
                concave_activation(x - c) + cc,
            )

        return convex, concave_activation, saturated_activation

    def apply_monotonicity_indicator_to_kernel(self, kernel):
        abs_kernel = torch.abs(kernel)
        kernel = torch.where(self.monotonicity_indicator == 1, abs_kernel, kernel)
        kernel = torch.where(self.monotonicity_indicator == -1, -abs_kernel, kernel)
        return kernel

    def apply_activations(self, h):
        if self.is_convex and self.is_concave:
            raise ValueError("The model cannot be set to be both convex and concave (only linear functions are both).")

        if self.is_convex:
            normalized_activation_weights = torch.tensor([1.0, 0.0, 0.0])
        elif self.is_concave:
            normalized_activation_weights = torch.tensor([0.0, 1.0, 0.0])
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
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int,
                 activation: str = 'elu', monotonicity_indicator: List[int] = None):
        super(MonotonicSequential, self).__init__()
        self.layers = nn.ModuleList()

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
            monotonicity_indicator=1  # Always increasing for output layer
        ))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# Example usage:
model = MonotonicSequential(
    input_size=3,
    hidden_sizes=[128, 128],
    output_size=1,
    activation='elu',
    monotonicity_indicator=[1, 0, -1]
)

print(sum(p.numel() for p in model.parameters() if p.requires_grad))

# Convert to PyTorch tensors
x_train2 = torch.tensor(x_train, dtype=torch.float32)
y_train2 = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
x_val2 = torch.tensor(x_val, dtype=torch.float32)
y_val2 = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

# Define optimizer and loss function
initial_lr = 0.005
lr_schedule = lambda epoch: initial_lr * (0.9 ** (epoch // (5000 // 16)))
optimizer = optim.Adam(model.parameters(), lr=initial_lr)
criterion = nn.MSELoss()

# Training loop
num_epochs = 20
batch_size = 16

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for i in range(0, len(x_train2), batch_size):
        batch_x = x_train2[i:i + batch_size]
        batch_y = y_train2[i:i + batch_size]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / (len(x_train2) / batch_size)

    # Update learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_schedule(epoch)

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(x_val2)
        val_loss = criterion(val_outputs, y_val2)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {val_loss.item():.4f}')
