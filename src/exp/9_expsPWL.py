# expsPWL

import csv
import numpy as np
import multiprocessing as mp
from functools import partial
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, accuracy_score
from typing import Callable, Tuple, List, Dict
from schedulefree import AdamWScheduleFree
from src.MLP import StandardMLP
from src.PWLNetwork import pwl
from dataPreprocessing.loaders import (load_abalone, load_auto_mpg, load_blog_feedback, load_boston_housing,
                                       load_compas, load_era, load_esl, load_heart, load_lev, load_loan, load_swd)
import random
from src.utils import monotonicity_check, get_reordered_monotonic_indices, write_results_to_csv, count_parameters
mp.set_start_method('spawn', force=True)
GLOBAL_SEED = 42

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_task_type(data_loader: Callable) -> str:
    regression_tasks = [load_abalone, load_auto_mpg, load_blog_feedback, load_boston_housing,
                        load_era, load_esl, load_lev, load_swd]
    return "regression" if data_loader in regression_tasks else "classification"

BEST_CONFIGS = {
    "load_abalone": {"lr": 0.04428367403666013, "hidden_sizes": [8, 64], "batch_size": 64, "epochs": 100},
    "load_auto_mpg": {"lr": 0.09811656676228442, "hidden_sizes": [32, 64], "batch_size": 16, "epochs": 100},
    "load_blog_feedback": {"lr": 0.020918413149522606, "hidden_sizes": [16, 16], "batch_size": 16, "epochs": 100},
    "load_boston_housing": {"lr": 0.0013983973589903281, "hidden_sizes": [64, 64], "batch_size": 16, "epochs": 100},
    "load_compas": {"lr": 0.006741349738796535, "hidden_sizes": [8, 64], "batch_size": 128, "epochs": 100},
    "load_era": {"lr": 0.0914834247268068, "hidden_sizes": [16, 32], "batch_size": 32, "epochs": 100},
    "load_esl": {"lr": 0.05661003724844464, "hidden_sizes": [32, 64], "batch_size": 64, "epochs": 100},
    "load_heart": {"lr": 0.09185139694044091, "hidden_sizes": [32, 16], "batch_size": 128, "epochs": 100},
    "load_lev": {"lr": 0.022946670310782055, "hidden_sizes": [32, 64], "batch_size": 16, "epochs": 100},
    "load_swd": {"lr": 0.07005056625746753, "hidden_sizes": [32, 64], "batch_size": 128, "epochs": 100},
    "load_loan": {"lr": 0.04249576921544568, "hidden_sizes": [16, 64], "batch_size": 32, "epochs": 100}
}


def create_model(config: Dict, input_size: int, task_type: str, seed: int) -> StandardMLP:
    torch.manual_seed(seed)
    output_activation = nn.Identity() if task_type == "regression" else nn.Sigmoid()
    return StandardMLP(
        input_size=input_size,
        hidden_sizes=config["hidden_sizes"],
        output_size=1,
        activation=nn.ReLU(),
        output_activation=output_activation
    )


def train_model(model: nn.Module, optimizer: AdamWScheduleFree, train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader, config: Dict, task_type: str,
                device: torch.device, monotonic_indices: List[int]) -> float:
    best_custom_loss = float('inf')
    patience = 30
    patience_counter = 0
    best_model_state = None
    max_epochs = config["epochs"]
    for epoch in range(max_epochs):
        model.train()
        optimizer.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            _ = pwl(model, optimizer, batch_X, batch_y, task_type, monotonic_indices,
                       config["monotonicity_weight"])

        model.eval()
        optimizer.eval()
        val_loss, custom_loss = evaluate_model(model, optimizer, val_loader, task_type, device,
                                               monotonic_indices, config["monotonicity_weight"])

        # Check for early stopping based on custom loss
        if custom_loss < best_custom_loss:
            best_custom_loss = custom_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(best_model_state)
    return best_custom_loss


@torch.no_grad()
def evaluate_model(model: nn.Module, optimizer: AdamWScheduleFree, data_loader: DataLoader,
                   task_type: str, device: torch.device, monotonic_indices: List[int],
                   monotonicity_weight: float) -> Tuple[float, float]:
    model.eval()
    optimizer.eval()
    predictions, true_values = [], []
    total_loss = 0.0
    criterion = nn.MSELoss() if task_type == "regression" else nn.BCELoss()

    for batch_X, batch_y in data_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        predictions.extend(outputs.cpu().numpy())
        true_values.extend(batch_y.cpu().numpy())

        # Calculate the empirical loss
        empirical_loss = criterion(outputs, batch_y)

        # Calculate monotonicity loss
        monotonicity_loss = torch.tensor(0.0, device=batch_X.device)
        if monotonic_indices:
            with torch.enable_grad():
                x_m = batch_X[:, monotonic_indices].requires_grad_(True)
                # Create a new input tensor with gradients only for monotonic features
                x_grad = batch_X.clone()
                x_grad[:, monotonic_indices] = x_m
                y_pred_m = model(x_grad)
                # Calculate gradients for each example with respect to monotonic features
                grads = torch.autograd.grad(y_pred_m.sum(), x_m, create_graph=True)[0]
                # Calculate divergence (sum of gradients across monotonic features)
                divergence = grads.sum(dim=1)
                # Apply max(0, -divergence) for each example
                monotonicity_term = torch.relu(-divergence)
                # Sum over all examples
                monotonicity_loss = monotonicity_term.sum()

        # Combine losses as per the equation in pwl
        batch_loss = empirical_loss + monotonicity_weight * monotonicity_loss
        total_loss += batch_loss.item()

    avg_total_loss = total_loss / len(data_loader)

    if task_type == "regression":
        validation_metric = np.sqrt(mean_squared_error(true_values, predictions))
    else:
        validation_metric = 1 - accuracy_score(np.squeeze(true_values), np.round(np.squeeze(predictions)))

    return validation_metric, avg_total_loss


# Add this new function to evaluate monotonicity
def evaluate_monotonicity(model: StandardMLP, optimizer: AdamWScheduleFree, train_loader: DataLoader,
                          val_loader: DataLoader,
                          device: torch.device, monotonic_indices: List[int]) -> Dict[str, float]:
    n_points = min(1000, len(val_loader.dataset))

    random_data = torch.rand(n_points, model.input_size, device=device)

    train_data = next(iter(train_loader))[0][:n_points].to(device)
    val_data = next(iter(val_loader))[0][:n_points].to(device)

    model.train()
    optimizer.train()
    mono_random = monotonicity_check(model, optimizer, random_data, monotonic_indices, device)

    model.train()
    optimizer.train()
    mono_train = monotonicity_check(model, optimizer, train_data, monotonic_indices, device)

    model.train()
    optimizer.train()
    mono_val = monotonicity_check(model, optimizer, val_data, monotonic_indices, device)

    model.eval()
    optimizer.eval()

    return {
        'random': mono_random,
        'train': mono_train,
        'val': mono_val
    }


def process_fold(fold, X, y, best_config, task_type, monotonic_indices, GLOBAL_SEED):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kf = KFold(n_splits=5, shuffle=True, random_state=GLOBAL_SEED)
    train_idx, val_idx = list(kf.split(X))[fold]
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    g = torch.Generator()
    g.manual_seed(GLOBAL_SEED + fold)

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).reshape(-1, 1))
    train_loader = DataLoader(train_dataset, batch_size=best_config["batch_size"], shuffle=True, generator=g)
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val).reshape(-1, 1))
    val_loader = DataLoader(val_dataset, batch_size=best_config["batch_size"], generator=g)

    model = create_model(best_config, X.shape[1], task_type, GLOBAL_SEED + fold).to(device)
    n_params = count_parameters(model)
    optimizer = AdamWScheduleFree(model.parameters(), lr=best_config["lr"], warmup_steps=5)
    _ = train_model(model, optimizer, train_loader, val_loader, best_config, task_type, device, monotonic_indices)

    val_metric, _ = evaluate_model(model, optimizer, val_loader, task_type, device, monotonic_indices,
                                   best_config["monotonicity_weight"])
    fold_mono_metrics = evaluate_monotonicity(model, optimizer, train_loader, val_loader, device, monotonic_indices)

    return val_metric, fold_mono_metrics, n_params


def cross_validate(X: np.ndarray, y: np.ndarray, best_config: Dict, task_type: str, monotonic_indices: List[int],
                   n_splits: int = 5) -> Tuple[List[float], Dict, int]:
    with mp.Pool(processes=min(mp.cpu_count(), 3)) as pool:
        results = pool.map(partial(process_fold, X=X, y=y, best_config=best_config, task_type=task_type,
                                   monotonic_indices=monotonic_indices, GLOBAL_SEED=GLOBAL_SEED),
                           range(n_splits))

    scores, mono_metrics_list, n_params_list = zip(*results)

    mono_metrics = {key: [] for key in mono_metrics_list[0].keys()}
    for fold_metrics in mono_metrics_list:
        for key in mono_metrics:
            mono_metrics[key].append(fold_metrics[key])

    return list(scores), mono_metrics, n_params_list[0]


def process_repeat(i, X_train, y_train, X_test, y_test, best_config, task_type, monotonic_indices, GLOBAL_SEED):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(GLOBAL_SEED + i)
    indices = np.random.permutation(len(X_train))
    X_train_shuffled, y_train_shuffled = X_train[indices], y_train[indices]

    g = torch.Generator()
    g.manual_seed(GLOBAL_SEED + i)

    train_dataset = TensorDataset(torch.FloatTensor(X_train_shuffled),
                                  torch.FloatTensor(y_train_shuffled).reshape(-1, 1))
    train_loader = DataLoader(train_dataset, batch_size=best_config["batch_size"], shuffle=True, generator=g)
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test).reshape(-1, 1))
    test_loader = DataLoader(test_dataset, batch_size=best_config["batch_size"], generator=g)

    model = create_model(best_config, X_train.shape[1], task_type, GLOBAL_SEED + i).to(device)
    n_params = count_parameters(model)
    optimizer = AdamWScheduleFree(model.parameters(), lr=best_config["lr"], warmup_steps=5)
    _ = train_model(model, optimizer, train_loader, test_loader, best_config, task_type, device, monotonic_indices)

    test_metric, _ = evaluate_model(model, optimizer, test_loader, task_type, device, monotonic_indices,
                                    best_config["monotonicity_weight"])
    fold_mono_metrics = evaluate_monotonicity(model, optimizer, train_loader, test_loader, device, monotonic_indices)

    return test_metric, fold_mono_metrics, n_params


def repeated_train_test(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                        best_config: Dict, task_type: str, monotonic_indices: List[int], n_repeats: int = 5) -> Tuple[
    List[float], Dict, int]:
    with mp.Pool(processes=min(mp.cpu_count(), 3)) as pool:
        results = pool.map(partial(process_repeat, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                   best_config=best_config, task_type=task_type, monotonic_indices=monotonic_indices,
                                   GLOBAL_SEED=GLOBAL_SEED),
                           range(n_repeats))

    scores, mono_metrics_list, n_params_list = zip(*results)

    mono_metrics = {key: [] for key in mono_metrics_list[0].keys()}
    for repeat_metrics in mono_metrics_list:
        for key in mono_metrics:
            mono_metrics[key].append(repeat_metrics[key])

    return list(scores), mono_metrics, n_params_list[0]

def process_dataset(data_loader: Callable, results_file: str) -> None:
    print(f"\nProcessing dataset: {data_loader.__name__}")
    X, y, X_test, y_test = data_loader()
    task_type = get_task_type(data_loader)
    monotonic_indices = get_reordered_monotonic_indices(data_loader.__name__)
    best_config = BEST_CONFIGS[data_loader.__name__]
    mono_weights = [0.1, 0.5, 1., 5., 10.]
    for weight in mono_weights:
        print(f"\nMonotonicity weight: {weight}")
        current_config = best_config.copy()
        current_config["monotonicity_weight"] = weight

        if data_loader == load_blog_feedback:
            scores, mono_metrics, n_params = repeated_train_test(X, y, X_test, y_test, current_config, task_type,
                                                                 monotonic_indices)
        else:
            X_combined = np.vstack((X, X_test))
            y_combined = np.concatenate((y, y_test))
            scores, mono_metrics, n_params = cross_validate(X_combined, y_combined, current_config, task_type,
                                                            monotonic_indices)

        avg_mono_metrics = {
            key: (np.mean(values), np.std(values)) for key, values in mono_metrics.items()
        }

        metric_name = "RMSE" if task_type == "regression" else "Error Rate"

        # Write results to CSV file
        write_results_to_csv(results_file, data_loader.__name__, task_type, metric_name,
                             np.mean(scores), np.std(scores), current_config, avg_mono_metrics, n_params)

        # Print results to console (optional)
        print(f"\nResults for {data_loader.__name__} with monotonicity weight {weight}:")
        print(f"{metric_name}: {np.mean(scores):.4f} (±{np.std(scores):.4f})")
        print(f"Configuration: {current_config}")
        print(f"Number of parameters: {n_params}")
        print("Monotonicity violation rates:")
        for key, (mean, std) in avg_mono_metrics.items():
            print(f"  {key.capitalize()} data: {mean:.4f} (±{std:.4f})")


def main():
    set_global_seed(GLOBAL_SEED)

    dataset_loaders = [
        load_abalone, load_auto_mpg, load_blog_feedback, load_boston_housing,
        load_compas, load_era, load_esl, load_heart, load_lev, load_swd, load_loan
    ]

    dataset_loaders = [
        load_loan
    ]

    results_file = "expsPWL.csv"

    # Create the CSV file and write the header
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Dataset", "Task Type", "Metric Name", "Metric Value", "Metric Std Dev", "NumofParameters",
            "Best Configuration",
            "Mono Random Mean", "Mono Random Std",
            "Mono Train Mean", "Mono Train Std",
            "Mono Val Mean", "Mono Val Std"
        ])

    for data_loader in dataset_loaders:
        process_dataset(data_loader, results_file)

if __name__ == '__main__':
    main()