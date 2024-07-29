import torch
import torch.nn as nn
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import List, Literal
import random
from itertools import combinations
from schedulefree import AdamWScheduleFree
from src.MLP import StandardMLP


class CertifiedMonotonicNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, monotonic_indices: List[int],
                 monotonicity_weight: float = 1.0, activation: nn.Module = nn.ReLU(),
                 output_activation: nn.Module = nn.Identity(), dropout_rate: float = 0.0, b: float = 0.2,
                 init_method: Literal[
                     'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'truncated_normal'] = 'xavier_uniform'):
        super().__init__()
        self.monotonicity_weight = monotonicity_weight
        self.b = b
        self.n_monotonic_features = len(monotonic_indices)
        self.n_non_monotonic_features = input_size - self.n_monotonic_features

        # Create a mask for monotonic and non-monotonic features
        self.monotonic_mask = torch.zeros(input_size, dtype=torch.bool)
        self.monotonic_mask[monotonic_indices] = True

        # Main network
        self.model = StandardMLP(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            activation=activation,
            output_activation=output_activation,
            dropout_rate=dropout_rate,
            init_method=init_method
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def uniform_pwl(model: nn.Module, optimizer: AdamWScheduleFree, x: torch.Tensor, y: torch.Tensor,
                       task_type: str, monotonic_indices: List[int], monotonicity_weight: float = 1.0,
                       regularization_budget: int = 1000, b:float = 0.2) -> torch.Tensor:
    device = x.device

    # Compute empirical loss
    y_pred = model(x)
    if task_type == "regression":
        empirical_loss = nn.MSELoss()(y_pred, y)
    else:
        empirical_loss = nn.BCELoss()(y_pred, y)

    # Prepare for regularization
    model.train()
    optimizer.train()

    # Generate regularization points
    regularization_points = torch.rand(regularization_budget, x.shape[1], device=device)

    # Separate monotonic and non-monotonic features
    monotonic_mask = torch.zeros(regularization_points.shape[1], dtype=torch.bool)
    monotonic_mask[monotonic_indices] = True
    data_monotonic = regularization_points[:, monotonic_mask]
    data_non_monotonic = regularization_points[:, ~monotonic_mask]

    data_monotonic.requires_grad_(True)

    def closure():
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(torch.cat([data_monotonic, data_non_monotonic], dim=1))
            loss = torch.sum(outputs)
        loss.backward()
        return loss

    closure()
    grad_wrt_monotonic_input = data_monotonic.grad

    if grad_wrt_monotonic_input is None:
        print("Warning: Gradient is None. Check if the model is correctly set up for gradient computation.")
        return empirical_loss

    # Compute regularization
    grad_penalty = torch.relu(-grad_wrt_monotonic_input + b) ** 2
    regularization = torch.max(torch.sum(grad_penalty, dim=1))

    return empirical_loss + monotonicity_weight * regularization


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
    for i in range(0, len(model.main_network.layers) - 1):
        first_layer = model.main_network.layers[i]
        second_layer = model.main_network.layers[i + 1]
        mono_flag = certify_grad_with_gurobi(first_layer, second_layer, model.n_monotonic_features)
        if not mono_flag:
            break
    return mono_flag

