# expsMinMax

import multiprocessing as mp
from functools import partial
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, accuracy_score
from typing import Callable, Tuple, List, Dict, Union
import optuna
from schedulefree import AdamWScheduleFree
from src.MinMaxNetwork import MinMaxNetwork, MinMaxNetworkWithMLP, SmoothMinMaxNetwork, SmoothMinMaxNetworkWithMLP
from dataPreprocessing.loaders import (load_abalone, load_auto_mpg, load_blog_feedback, load_boston_housing,
                                       load_compas, load_era, load_esl, load_heart, load_lev, load_loan, load_swd)
import random
from src.utils import monotonicity_check, get_reordered_monotonic_indices, write_results_to_csv, count_parameters

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


def create_model(config: Dict, input_size: int, task_type: str, seed: int, monotonic_indices: List[int],
                 model_type: str, device: torch.device) -> nn.Module:
    torch.manual_seed(seed)

    if model_type == "minmax":
        return MinMaxNetwork(
            input_size=input_size,
            K=config["K"],
            h_K=config["h_K"],
            monotonic_indices=monotonic_indices,
            device=device
        )
    elif model_type == "minmax_aux":
        return MinMaxNetworkWithMLP(
            input_size=input_size,
            K=config["K"],
            h_K=config["h_K"],
            monotonic_indices=monotonic_indices,
            aux_hidden_units=config["aux_hidden_units"],
            device=device
        )
    elif model_type == "smooth_minmax":
        return SmoothMinMaxNetwork(
            input_size=input_size,
            K=config["K"],
            h_K=config["h_K"],
            monotonic_indices=monotonic_indices,
            beta=config["beta"],
            device=device
        )
    elif model_type == "smooth_minmax_aux":
        return SmoothMinMaxNetworkWithMLP(
            input_size=input_size,
            K=config["K"],
            h_K=config["h_K"],
            monotonic_indices=monotonic_indices,
            aux_hidden_units=config["aux_hidden_units"],
            beta=config["beta"],
            device=device
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def train_model(model: nn.Module, optimizer, train_loader: DataLoader, val_loader: DataLoader,
                config: Dict, task_type: str, device: torch.device) -> float:
    criterion = nn.MSELoss() if task_type == "regression" else nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    best_model_state = None

    for epoch in range(config["epochs"]):
        model.train()
        optimizer.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            def closure():
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                return loss

            optimizer.step(closure)

        model.eval()
        optimizer.eval()
        val_loss = evaluate_model(model, optimizer, val_loader, task_type, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            # print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_model_state)
    return best_val_loss

@torch.no_grad()
def evaluate_model(model: nn.Module, optimizer: AdamWScheduleFree, data_loader: DataLoader,
                   task_type: str, device: torch.device) -> float:
    model.eval()
    optimizer.eval()
    predictions, true_values = [], []
    for batch_X, batch_y in data_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        predictions.extend(outputs.cpu().numpy())
        true_values.extend(batch_y.cpu().numpy())

    if task_type == "regression":
        return np.sqrt(mean_squared_error(true_values, predictions))
    else:
        predictions_array = np.array(predictions)
        binary_preds = (torch.sigmoid(torch.tensor(predictions_array)) > 0.5).numpy()
        return 1 - accuracy_score(np.squeeze(true_values), binary_preds)

def optimize_hyperparameters(X: np.ndarray, y: np.ndarray, task_type: str, monotonic_indices: List[int],
                             model_type: str, n_trials: int = 30, sample_size: int = 50000) -> Dict[
    str, Union[float, List[int], int]]:
    if len(X) > sample_size:
        np.random.seed(GLOBAL_SEED)
        indices = np.random.choice(len(X), sample_size, replace=False)
        X_sampled, y_sampled = X[indices], y[indices]
    else:
        X_sampled, y_sampled = X, y

    dataset = TensorDataset(torch.FloatTensor(X_sampled), torch.FloatTensor(y_sampled).reshape(-1, 1))
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(GLOBAL_SEED))

    def objective(trial):
        config = {
            "lr": trial.suggest_float("lr", 1e-3, 1e-1, log=True),
            "K": trial.suggest_int("K", 2, 6),
            "h_K": trial.suggest_categorical("h_K", [4, 8, 16, 32, 64]),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
            "epochs": 100,
        }

        if "aux" in model_type:
            config["aux_hidden_units"] = trial.suggest_categorical("aux_hidden_units", [8, 16, 32, 64])

        if "smooth" in model_type:
            config["beta"] = trial.suggest_float("beta", -2.0, 2.0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        g = torch.Generator()
        g.manual_seed(GLOBAL_SEED)

        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, generator=g)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], generator=g)

        input_size = dataset.tensors[0].shape[1]
        model = create_model(config, input_size, task_type, GLOBAL_SEED, monotonic_indices, model_type, device).to(device)
        optimizer = AdamWScheduleFree(model.parameters(), lr=config["lr"], warmup_steps=5)
        val_metric = train_model(model, optimizer, train_loader, val_loader, config, task_type, device)

        return val_metric

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=GLOBAL_SEED))

    try:
        #n_jobs = max(1, multiprocessing.cpu_count() // 2)
        n_jobs = -1
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False, n_jobs=n_jobs)
        best_params = study.best_params
        best_params["epochs"] = 100
    except ValueError as e:
        print(f"Optimization failed: {e}")
        # Return default parameters if optimization fails
        best_params = {
            "lr": 0.001,
            "K": 5,
            "h_K": 32,
            "batch_size": 32,
            "epochs": 100
        }
        if "aux" in model_type:
            best_params["aux_hidden_units"] = 64
        if "smooth" in model_type:
            best_params["beta"] = 0.0

    return best_params


def evaluate_with_monotonicity(model: nn.Module, optimizer, train_loader: DataLoader, val_loader: DataLoader,
                               task_type: str, device: torch.device, monotonic_indices: List[int]) -> Tuple[float, Dict]:
    model.eval()
    optimizer.eval()
    predictions, true_values = [], []

    # Collect data for monotonicity check
    train_data, train_preds = [], []
    val_data, val_preds = [], []

    with torch.no_grad():
        for loader, data_list, pred_list in [(train_loader, train_data, train_preds),
                                             (val_loader, val_data, val_preds)]:
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)

                data_list.append(batch_X.cpu())
                pred_list.append(outputs.cpu())

                if loader == val_loader:
                    predictions.extend(outputs.cpu().numpy())
                    true_values.extend(batch_y.cpu().numpy())

    # Concatenate data for monotonicity check
    train_data = torch.cat(train_data)
    val_data = torch.cat(val_data)

    if task_type == "regression":
        metric = np.sqrt(mean_squared_error(true_values, predictions))
    else:
        predictions_array = np.array(predictions)
        binary_preds = (torch.sigmoid(torch.tensor(predictions_array)) > 0.5).numpy()
        metric = 1 - accuracy_score(np.squeeze(true_values), binary_preds)

    n_points = min(1000, len(val_loader.dataset))

    # Generate random data for monotonicity check
    random_data = torch.rand(n_points, model.input_size, device=device)

    model.train()
    optimizer.train()
    mono_random = monotonicity_check(model, optimizer, random_data, monotonic_indices, device)

    model.train()
    optimizer.train()
    mono_train = monotonicity_check(model, optimizer, train_data[:n_points].to(device),
                                    monotonic_indices, device)

    model.train()
    optimizer.train()
    mono_val = monotonicity_check(model, optimizer, val_data[:n_points].to(device),
                                  monotonic_indices, device)

    model.eval()
    optimizer.eval()

    mono_metrics = {
        'random': mono_random,
        'train': mono_train,
        'val': mono_val
    }

    return metric, mono_metrics

def process_fold(fold_data, best_config, task_type, monotonic_indices, model_type, GLOBAL_SEED):
    fold, (train_idx, val_idx), X, y = fold_data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Use a deterministic generator
    g = torch.Generator()
    g.manual_seed(GLOBAL_SEED)

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).reshape(-1, 1))
    train_loader = DataLoader(train_dataset, batch_size=best_config["batch_size"], shuffle=True, generator=g)
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val).reshape(-1, 1))
    val_loader = DataLoader(val_dataset, batch_size=best_config["batch_size"], generator=g)

    model = create_model(best_config, X.shape[1], task_type, GLOBAL_SEED, monotonic_indices, model_type, device).to(
        device)
    n_params = count_parameters(model)
    optimizer = AdamWScheduleFree(model.parameters(), lr=best_config["lr"], warmup_steps=5)
    _ = train_model(model, optimizer, train_loader, val_loader, best_config, task_type, device)

    val_metric, fold_mono_metrics = evaluate_with_monotonicity(model, optimizer, train_loader, val_loader, task_type,
                                                               device, monotonic_indices)

    return val_metric, fold_mono_metrics, n_params


def process_repeat(repeat_data, best_config, task_type, monotonic_indices, model_type, GLOBAL_SEED):
    i, X_train, y_train, X_test, y_test = repeat_data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use numpy's RandomState for reproducibility
    rng = np.random.RandomState(GLOBAL_SEED)
    indices = rng.permutation(len(X_train))
    X_train_shuffled, y_train_shuffled = X_train[indices], y_train[indices]

    g = torch.Generator()
    g.manual_seed(GLOBAL_SEED)

    train_dataset = TensorDataset(torch.FloatTensor(X_train_shuffled),
                                  torch.FloatTensor(y_train_shuffled).reshape(-1, 1))
    train_loader = DataLoader(train_dataset, batch_size=best_config["batch_size"], shuffle=True, generator=g)
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test).reshape(-1, 1))
    test_loader = DataLoader(test_dataset, batch_size=best_config["batch_size"], generator=g)

    model = create_model(best_config, X_train.shape[1], task_type, GLOBAL_SEED, monotonic_indices, model_type,
                         device).to(device)
    n_params = count_parameters(model)
    optimizer = AdamWScheduleFree(model.parameters(), lr=best_config["lr"], warmup_steps=5)
    _ = train_model(model, optimizer, train_loader, test_loader, best_config, task_type, device)

    test_metric, fold_mono_metrics = evaluate_with_monotonicity(model, optimizer, train_loader, test_loader, task_type,
                                                                device, monotonic_indices)

    return test_metric, fold_mono_metrics, n_params


def cross_validate(X: np.ndarray, y: np.ndarray, best_config: Dict, task_type: str, monotonic_indices: List[int],
                   model_type: str, n_splits: int = 5) -> Tuple[List[float], Dict, int]:
    # Create folds with a fixed random state
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=GLOBAL_SEED)
    folds = list(kf.split(X))

    # Prepare data for each fold
    fold_data = [(fold, split, X, y) for fold, split in enumerate(folds)]

    with mp.Pool(processes=min(mp.cpu_count(), 3)) as pool:
        results = pool.map(partial(process_fold, best_config=best_config, task_type=task_type,
                                   monotonic_indices=monotonic_indices, model_type=model_type, GLOBAL_SEED=GLOBAL_SEED),
                           fold_data)

    scores, mono_metrics_list, n_params_list = zip(*results)

    mono_metrics = {key: [] for key in mono_metrics_list[0].keys()}
    for fold_metrics in mono_metrics_list:
        for key in mono_metrics:
            mono_metrics[key].append(fold_metrics[key])

    return list(scores), mono_metrics, n_params_list[0]

def repeated_train_test(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                        best_config: Dict, task_type: str, monotonic_indices: List[int], model_type: str,
                        n_repeats: int = 5) -> Tuple[
    List[float], Dict, int]:
    # Prepare data for each repeat
    repeat_data = [(i, X_train, y_train, X_test, y_test) for i in range(n_repeats)]

    with mp.Pool(processes=min(mp.cpu_count(), 3)) as pool:
        results = pool.map(partial(process_repeat, best_config=best_config, task_type=task_type,
                                   monotonic_indices=monotonic_indices, model_type=model_type, GLOBAL_SEED=GLOBAL_SEED),
                           repeat_data)

    scores, mono_metrics_list, n_params_list = zip(*results)

    mono_metrics = {key: [] for key in mono_metrics_list[0].keys()}
    for repeat_metrics in mono_metrics_list:
        for key in mono_metrics:
            mono_metrics[key].append(repeat_metrics[key])

    return list(scores), mono_metrics, n_params_list[0]


def process_dataset(data_loader: Callable, model_type: str, sample_size: int = 50000) -> Tuple[List[float], Dict, Dict, int]:
    print(f"\nProcessing dataset: {data_loader.__name__} with model: {model_type}")
    X, y, X_test, y_test = data_loader()
    task_type = get_task_type(data_loader)
    monotonic_indices = get_reordered_monotonic_indices(data_loader.__name__)
    n_trials = 50
    best_config = optimize_hyperparameters(X, y, task_type, monotonic_indices, model_type, sample_size=sample_size, n_trials=n_trials)

    if data_loader == load_blog_feedback:
        scores, mono_metrics, n_params = repeated_train_test(X, y, X_test, y_test, best_config, task_type, monotonic_indices, model_type)
    else:
        X = np.vstack((X, X_test))
        y = np.concatenate((y, y_test))
        scores, mono_metrics, n_params = cross_validate(X, y, best_config, task_type, monotonic_indices, model_type)

    avg_mono_metrics = {
        key: (np.mean(values), np.std(values)) for key, values in mono_metrics.items()
    }

    return scores, best_config, avg_mono_metrics, n_params


def main():
    set_global_seed(GLOBAL_SEED)
    mp.set_start_method('spawn', force=True)

    dataset_loaders = [
        load_abalone, load_auto_mpg, load_blog_feedback, load_boston_housing,
        load_compas, load_era, load_esl, load_heart, load_lev, load_swd, load_loan
    ]

    dataset_loaders = [
        load_loan
    ]

    model_types = ["minmax", "minmax_aux", "smooth_minmax", "smooth_minmax_aux"]
    model_types = ["minmax_aux"]

    sample_size = 40000

    for model_type in model_types:
        results_file = f"exps_{model_type}.csv"

        # Create the CSV file and write the header
        with open(results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Dataset", "Task Type", "Metric Name", "Metric Value", "Metric Std Dev", "NumOfParameters",
                "Best Configuration",
                "Mono Random Mean", "Mono Random Std",
                "Mono Train Mean", "Mono Train Std",
                "Mono Val Mean", "Mono Val Std"
            ])

        for data_loader in dataset_loaders:
            task_type = get_task_type(data_loader)
            scores, best_config, mono_metrics, n_params = process_dataset(data_loader, model_type, sample_size)
            metric_name = "RMSE" if task_type == "regression" else "Error Rate"

            # Write results to CSV file
            write_results_to_csv(results_file, data_loader.__name__, task_type, metric_name,
                                 np.mean(scores), np.std(scores), best_config, mono_metrics, n_params)

            # Print results to console (optional, you can remove this if you only want file output)
            print(f"\nResults for {data_loader.__name__} with {model_type}:")
            print(f"{metric_name}: {np.mean(scores):.4f} (±{np.std(scores):.4f})")
            print(f"Best configuration: {best_config}")
            print(f"Number of parameters: {n_params}")
            print("Monotonicity violation rates:")
            for key, (mean, std) in mono_metrics.items():
                print(f"  {key.capitalize()} data: {mean:.4f} (±{std:.4f})")

if __name__ == '__main__':
    main()