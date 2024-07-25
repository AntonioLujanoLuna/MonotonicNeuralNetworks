import torch
import torch.nn as nn
from typing import List, Literal
import random
from itertools import combinations
from MLP import StandardMLP
from torch.utils.data import DataLoader

class MixupRegularizerNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int,
                 monotonic_indices: List[int], monotonicity_weight: float = 1.0,
                 regularization_type: str = 'mixup',
                 init_method: Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'he_uniform', 'he_normal', 'truncated_normal'] = 'xavier_uniform'
):
        super(MixupRegularizerNetwork, self).__init__()
        self.monotonic_indices = monotonic_indices
        self.monotonicity_weight = monotonicity_weight
        self.regularization_type = regularization_type
        self.n_monotonic_features = len(monotonic_indices)
        self.init_method = init_method

        self.network = StandardMLP(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            activation=nn.ReLU(),
            output_activation=nn.Identity(),
            init_method=self.init_method
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def compute_mixup_regularizer(self, data_x: torch.Tensor, regularization_budget: int):
        self.eval()
        device = data_x.device

        if self.regularization_type == 'random':
            regularization_points = torch.rand(regularization_budget, data_x.shape[1], device=device)
        elif self.regularization_type == 'train':
            if regularization_budget > data_x.shape[0]:
                regularization_points = data_x.repeat(regularization_budget // data_x.shape[0] + 1, 1)[
                                        :regularization_budget]
            else:
                regularization_points = data_x[:regularization_budget]
        else:  # mixup
            random_data = torch.rand_like(data_x)
            combined_data = torch.cat([data_x, random_data], dim=0)
            pairs = self.get_pairs(combined_data, max_n_pairs=regularization_budget)
            regularization_points = self.interpolate_pairs(pairs)

        regularization_points_monotonic = regularization_points[:, self.monotonic_indices]
        regularization_points_monotonic.requires_grad = True

        predictions = self(regularization_points)

        grad_wrt_monotonic_input = torch.autograd.grad(
            torch.sum(predictions),
            regularization_points_monotonic,
            create_graph=True,
            allow_unused=True,
        )[0]

        grad_penalty = torch.relu(-grad_wrt_monotonic_input) ** 2
        regularization = torch.mean(torch.sum(grad_penalty, dim=1))

        self.train()
        return regularization

    def get_pairs(self, data, max_n_pairs):
        all_pairs = list(combinations(range(len(data)), 2))
        if len(all_pairs) > max_n_pairs:
            all_pairs = random.sample(all_pairs, max_n_pairs)
        all_pairs = torch.LongTensor(all_pairs).to(data.device)

        pairs_left = torch.index_select(data, 0, all_pairs[:, 0])
        pairs_right = torch.index_select(data, 0, all_pairs[:, 1])

        return pairs_left, pairs_right

    def interpolate_pairs(self, pairs, interpolation_range=0.5):
        pairs_left, pairs_right = pairs
        lower_bound = 0.5 - interpolation_range
        upper_bound = 0.5 + interpolation_range
        interpolation_factors = torch.rand(len(pairs_left), 1, device=pairs_left.device) * (
                    upper_bound - lower_bound) + lower_bound

        return interpolation_factors * pairs_left + (1 - interpolation_factors) * pairs_right

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor, loss_fn: nn.Module,
                     regularization_budget: int) -> torch.Tensor:
        y_pred = self(x)
        empirical_loss = loss_fn(y_pred, y)

        regularization_loss = self.compute_mixup_regularizer(x, regularization_budget)

        return empirical_loss + self.monotonicity_weight * regularization_loss


def train_mixup_regularizer_network(model: MixupRegularizerNetwork, train_loader: DataLoader,
                                    optimizer: torch.optim.Optimizer, loss_fn: nn.Module, num_epochs: int,
                                    regularization_budget: int):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = model.compute_loss(batch_x, batch_y, loss_fn, regularization_budget)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")

# Example usage:
# input_size = 10
# hidden_sizes = [64, 32]
# output_size = 1
# monotonic_indices = [0, 2]  # Features 0 and 2 are monotonically increasing
# model = MixupRegularizerNetwork(input_size, hidden_sizes, output_size, monotonic_indices, regularization_type='mixup')
# optimizer = torch.optim.Adam(model.parameters())
# loss_fn = nn.MSELoss()
# train_loader = torch.utils.data.DataLoader(your_dataset, batch_size=32, shuffle=True)
# train_mixup_regularizer_network(model, train_loader, optimizer, loss_fn, num_epochs=100, regularization_budget=1000)