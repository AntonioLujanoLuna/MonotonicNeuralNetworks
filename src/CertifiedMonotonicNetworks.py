import torch
import torch.nn as nn
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import List


def uniformPWL_mono_reg(model: nn.Module, x: torch.Tensor, monotonic_indices: List[int], b: float = 0.2):
    x_m = x[:, monotonic_indices]
    x_m.requires_grad_(True)

    x_grad = x.clone()
    x_grad[:, monotonic_indices] = x_m
    y_pred_m = model(x_grad)

    grads = torch.autograd.grad(y_pred_m.sum(), x_m, create_graph=True, allow_unused=True)[0]
    divergence = grads.sum(dim=1)
    monotonicity_term = torch.relu(-divergence + b) ** 2
    monotonicity_loss = monotonicity_term.max()
    return monotonicity_loss


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

