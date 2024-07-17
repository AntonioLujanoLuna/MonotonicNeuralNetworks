import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple
from pmlayer.torch.layers import HLattice
from MLP import StandardMLP

class Progress:
    """
    Training Progress tracker.

    Implementation follows:
    Prechelt, Lutz. Early Stopping—But When? Neural Networks: Tricks of the Trade (2012): 53-67.

    Attributes:
        strip (int): Length of training strip observed.
        threshold (float): Lower threshold on progress.
        E (np.ndarray): Array to store recent error values.
        t (int): Current time step.
        valid (bool): Whether enough data has been collected to make a valid decision.
    """

    def __init__(self, strip: int = 5, threshold: float = 0.01):
        self.strip = strip
        self.E = np.ones(strip)
        self.t = 0
        self.valid = False
        self.threshold = threshold

    def progress(self) -> float:
        """Calculate the current progress."""
        return 1000 * ((self.E.mean() / self.E.min()) - 1.)

    def stop(self) -> bool:
        """Determine if training should stop based on progress."""
        if not self.valid:
            return False
        return self.progress() < self.threshold

    def update(self, e: float) -> bool:
        """
        Update the progress tracker with a new error value.

        Args:
            e (float): New error value.

        Returns:
            bool: Whether training should stop.
        """
        self.E[self.t % self.strip] = e
        self.t += 1
        if self.t >= self.strip:
            self.valid = True
        return self.stop()


def total_params(model: nn.Module) -> int:
    """
    Compute the total number of parameters in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        int: Total number of parameters.
    """
    return sum(param.numel() for param in model.parameters())


def fit_torch(
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        threshold: float = 1e-3,
        max_iterations: int = 100000
) -> None:
    """
    Gradient-based fitting of PyTorch models using RProp.

    Args:
        model (nn.Module): The model to fit.
        x (torch.Tensor): Input tensor.
        y (torch.Tensor): Target tensor.
        threshold (float): Threshold on learning progress.
        max_iterations (int): Maximum number of iterations.
    """
    P = Progress(5, threshold=threshold)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Rprop(model.parameters(), lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))

    for epoch in range(max_iterations):
        pred_y = model(x)
        loss = loss_function(pred_y, y)
        if P.update(loss.item()):
            return
        loss.backward()
        optimizer.step()
        model.zero_grad()


def fit_torch_val(
        model: nn.Module,
        best_model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
        threshold: int = 100,
        max_iterations: int = 100000,
        verbose: int = 1
) -> None:
    """
    Gradient-based fitting of PyTorch models using RProp with validation data.

    Args:
        model (nn.Module): The model to fit.
        best_model (nn.Module): The model to store the best weights.
        x (torch.Tensor): Input training tensor.
        y (torch.Tensor): Target training tensor.
        x_val (torch.Tensor): Input validation tensor.
        y_val (torch.Tensor): Target validation tensor.
        threshold (int): Threshold on number of iterations without improvement.
        max_iterations (int): Maximum number of iterations.
        verbose (int): If not 0, report if training was stopped due to lack of progress.
    """
    best_val = float('inf')
    best_epoch = 0
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Rprop(model.parameters(), lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))

    for epoch in range(max_iterations):
        pred_y = model(x).reshape(-1, 1)
        loss = loss_function(pred_y, y)
        loss.backward()
        optimizer.step()
        model.zero_grad()

        with torch.no_grad():
            pred_y_val = model(x_val).reshape(-1, 1)
            loss_val = loss_function(pred_y_val, y_val)

        if loss_val < best_val:
            best_val = loss_val
            best_model.load_state_dict(model.state_dict())
            best_epoch = epoch

        if epoch - best_epoch > threshold:
            if verbose:
                print(f"({epoch}) ", end='')
            break


def train_hlattice(
        dim: int,
        lattice_sizes: List[int],
        increasing: List[int],
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        mlp_neurons: List[int],
        maxiter: int = 100000,
        clip: bool = False
) -> Tuple[float, float, int]:
    """
    Train a Hierarchical Lattice (HLattice) model.

    Args:
        dim (int): Input dimension.
        lattice_sizes (List[int]): Sizes of lattices at each level.
        increasing (List[int]): Indices of increasing dimensions.
        x_train (np.ndarray): Training input data.
        y_train (np.ndarray): Training target data.
        x_val (np.ndarray): Validation input data.
        y_val (np.ndarray): Validation target data.
        x_test (np.ndarray): Test input data.
        y_test (np.ndarray): Test target data.
        mlp_neurons (List[int]): List of neuron counts for hidden layers in MLP.
        maxiter (int): Maximum number of training iterations.
        clip (bool): Whether to clip output to [0, 1] range.

    Returns:
        Tuple[float, float, int]: Training MSE, Test MSE, and number of model parameters.
    """
    lattice_sizes_tensor = torch.tensor(lattice_sizes)
    x_train_torch = torch.FloatTensor(x_train)
    y_train_torch = torch.FloatTensor(y_train)
    x_val_torch = torch.FloatTensor(x_val)
    y_val_torch = torch.FloatTensor(y_val)
    x_test_torch = torch.FloatTensor(x_test)

    input_len = dim - len(increasing)
    output_len = torch.prod(lattice_sizes_tensor).item()

    ann = StandardMLP(input_len, mlp_neurons, output_len)
    model = HLattice(dim, lattice_sizes_tensor, increasing, ann)

    ann_best = StandardMLP(input_len, mlp_neurons, output_len)
    model_best = HLattice(dim, lattice_sizes_tensor, increasing, ann_best)

    fit_torch_val(model, model_best, x_train_torch, y_train_torch, x_val_torch, y_val_torch, max_iterations=maxiter)

    with torch.no_grad():
        y_hll_train = model_best(x_train_torch).numpy().reshape(-1, 1)
        y_hll_test = model_best(x_test_torch).numpy().reshape(-1, 1)

    mse_train = np.mean((y_train - y_hll_train) ** 2)
    mse_test = np.mean((y_test - (np.clip(y_hll_test, 0., 1.) if clip else y_hll_test)) ** 2)

    num_params = total_params(model)

    print(f"Test MSE: {mse_test}")

    return mse_train, mse_test, num_params