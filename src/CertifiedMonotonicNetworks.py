import torch
import torch.nn as nn
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import List
import torch.utils.data as Data
import random
from itertools import combinations


class CertifiedMonotonicNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int,
                 monotonic_indices: List[int], monotonicity_weight: float = 1.0,
                 regularization_type: str = 'random', b: float = 0.2):
        super(CertifiedMonotonicNetwork, self).__init__()
        self.monotonic_indices = monotonic_indices
        self.monotonicity_weight = monotonicity_weight
        self.regularization_type = regularization_type
        self.n_monotonic_features = len(monotonic_indices)
        self.b = b

        # Enforce architecture: alternating Linear and ReLU layers
        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

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

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor, loss_fn: nn.Module) -> torch.Tensor:
        y_pred = self(x)
        empirical_loss = loss_fn(y_pred, y)
        regularization_loss = self.compute_mixup_regularizer(x, regularization_budget=1024)
        return empirical_loss + self.monotonicity_weight * regularization_loss


def certify_grad_with_gurobi(first_layer, second_layer, mono_feature_num, direction=None):
    mono_flag = True
    w1 = first_layer.weight.data.detach().cpu().numpy().astype('float64')
    w2 = second_layer.weight.data.detach().cpu().numpy().astype('float64')
    b1 = first_layer.bias.data.detach().cpu().numpy().astype('float64')
    b2 = second_layer.bias.data.detach().cpu().numpy().astype('float64')
    feature_num = w1.shape[1]

    for p in range(mono_feature_num):
        if direction is not None:
            if direction[p] == -1:
                w2 = -w2

        fc_first = w1[:, p]

        m_up = np.sum(np.maximum(w1, 0.0), axis=1) + b1
        m_down = -np.sum(np.maximum(-w1, 0.0), axis=1) + b1
        h = np.concatenate((-b1, b1 - m_down), axis=0)

        G_z = np.zeros((w1.shape[0] * 2, w1.shape[0]))
        G_x = np.zeros((w1.shape[0] * 2, feature_num))
        for i in range(w1.shape[0]):
            G_x[i, :] = w1[i, :]
            G_z[i, i] = -m_up[i]

        for i in range(w1.shape[0]):
            G_x[i + w1.shape[0], :] = -w1[i, :]
            G_z[i + w1.shape[0], i] = -m_down[i]

        m = gp.Model("matrix1")
        m.Params.OutputFlag = 0
        z = m.addMVar(shape=w1.shape[0], vtype=GRB.BINARY, name="z")
        a = m.addMVar(shape=w2.shape[0], lb=0.0, vtype=GRB.CONTINUOUS, name="a")
        x = m.addMVar(shape=feature_num, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")

        obj_mat = np.zeros((w2.shape[0], w1.shape[0]))
        for q in range(w2.shape[0]):
            fc_last = w2[q, :]
            c = fc_last * fc_first
            obj_mat[q, :] = c

        one_array = np.ones((w2.shape[0]))
        m.addConstr(one_array.T @ a == 1., name="constraint_a")
        m.addConstr((G_z @ z + G_x @ x) <= h, name="constraint")
        m.setObjective(a @ (obj_mat @ z), GRB.MINIMIZE)
        m.optimize()

        if m.objVal < 0.:
            print('Non-monotonic')
            mono_flag = False
            break

        if direction is not None:
            if direction[p] == -1:
                w2 = -w2

    return mono_flag


def certify_monotonicity(model: CertifiedMonotonicNetwork):
    mono_flag = True
    for i in range(0, len(model.network) - 2, 2):  # Check pairs of linear layers
        first_layer = model.network[i]
        second_layer = model.network[i + 2]
        mono_flag = certify_grad_with_gurobi(first_layer, second_layer, model.n_monotonic_features)
        if not mono_flag:
            break
    return mono_flag


def train_certified_monotonic_network(model: CertifiedMonotonicNetwork, train_loader: Data.DataLoader,
                                      optimizer: torch.optim.Optimizer, loss_fn: nn.Module, num_epochs: int,
                                      initial_weight: float = 1.0, weight_increase_factor: float = 10.0):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            optimizer.zero_grad()
            loss = model.compute_loss(batch_x, batch_y, loss_fn)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")

        if (epoch + 1) % 10 == 0:
            if certify_monotonicity(model):
                print("Model certified as monotonic!")
                break
            else:
                model.monotonicity_weight *= weight_increase_factor
                print(f"Increased monotonicity weight to {model.monotonicity_weight}")

    return model

# Example usage:
# input_size = 10
# hidden_sizes = [64, 32]
# output_size = 1
# monotonic_indices = [0, 2]  # Features 0 and 2 are monotonically increasing
# model = CertifiedMonotonicNetwork(input_size, hidden_sizes, output_size, monotonic_indices)
# optimizer = torch.optim.Adam(model.parameters())
# loss_fn = nn.BCEWithLogitsLoss()
# train_loader = torch.utils.data.DataLoader(your_dataset, batch_size=32, shuffle=True)
# trained_model = train_certified_monotonic_network(model, train_loader, optimizer, loss_fn, num_epochs=200)