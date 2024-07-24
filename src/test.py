import ast
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from typing import Callable, Tuple, List, Dict
import optuna
from schedulefree import AdamWScheduleFree
from MLP import StandardMLP
from dataPreprocessing.loaders import (load_abalone, load_auto_mpg, load_blog_feedback, load_boston_housing,
                                       load_compas, load_era, load_esl, load_heart, load_lev, load_loan, load_swd)
import random
from utils import monotonicity_check, get_monotonic_indices

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


def create_model(config: Dict, input_size: int, task_type: str, seed: int) -> StandardMLP:
    torch.manual_seed(seed)
    output_activation = nn.Identity() if task_type == "regression" else nn.Sigmoid()
    return StandardMLP(
        input_size=input_size,
        hidden_sizes=config["hidden_sizes"],
        output_size=1,
        activation=nn.ReLU(),
        output_activation=output_activation,
    )

def train_model(model: nn.Module, optimizer, train_loader: DataLoader, val_loader: DataLoader,
                config: Dict, task_type: str, device: torch.device) -> float:
    criterion = nn.MSELoss() if task_type == "regression" else nn.BCELoss()

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

def evaluate_model(model: nn.Module, optimizer, data_loader: DataLoader, task_type: str, device: torch.device) -> float:
    model.eval()
    optimizer.eval()
    predictions, true_values = [], []
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            predictions.extend(outputs.cpu().numpy())
            true_values.extend(batch_y.cpu().numpy())

    if task_type == "regression":
        return np.sqrt(mean_squared_error(true_values, predictions))
    else:
        return 1 - accuracy_score(np.squeeze(true_values), np.round(np.squeeze(predictions)))


def objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
              task_type: str) -> float:
    config = {
        "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
        "hidden_sizes": ast.literal_eval(trial.suggest_categorical("hidden_sizes",
                                                                   ["[8, 8]", "[8, 16]", "[16, 8]", "[16, 16]",
                                                                    "[16, 32]", "[32, 32]", "[32, 64]", "[64, 64]"])),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        "epochs": 100,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    g = torch.Generator()
    g.manual_seed(GLOBAL_SEED)

    train_dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y).reshape(-1, 1))
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, generator=g)
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val).reshape(-1, 1))
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], generator=g)

    model = create_model(config, X.shape[1], task_type, GLOBAL_SEED).to(device)
    optimizer = AdamWScheduleFree(model.parameters(), lr=config["lr"], warmup_steps=5)
    val_metric = train_model(model, optimizer, train_loader, val_loader, config, task_type, device)

    return val_metric


def optimize_hyperparameters(X: np.ndarray, y: np.ndarray, task_type: str, n_trials: int = 30,
                             sample_size: int = 50000) -> Dict:
    if len(X) > sample_size:
        if task_type == "classification":
            _, X_sampled, _, y_sampled = train_test_split(
                X, y,
                train_size=sample_size,
                stratify=y,
                random_state=GLOBAL_SEED
            )
        else:
            np.random.seed(GLOBAL_SEED)
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sampled, y_sampled = X[indices], y[indices]
    else:
        X_sampled, y_sampled = X, y

    X_train, X_val, y_train, y_val = train_test_split(
        X_sampled, y_sampled,
        test_size=0.2,
        stratify=y_sampled if task_type == "classification" else None,
        random_state=GLOBAL_SEED
    )

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=GLOBAL_SEED))

    try:
        study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, task_type),
                       n_trials=n_trials, show_progress_bar=False, n_jobs=-1)
        best_params = study.best_params
        best_params["epochs"] = 100
    except ValueError as e:
        print(f"Optimization failed: {e}")
        # Return default parameters if optimization fails
        best_params = {
            "lr": 0.001,
            "hidden_sizes": [32, 32],
            "batch_size": 32,
            "epochs": 100
        }

    return best_params


def evaluate_with_monotonicity(model: nn.Module, optimizer, train_loader: DataLoader, val_loader: DataLoader,
                               task_type: str, device: torch.device, monotonic_indices: List[int]) -> Tuple[
    float, Dict]:
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
        metric = 1 - accuracy_score(np.squeeze(true_values), np.round(np.squeeze(predictions)))

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

def cross_validate(X: np.ndarray, y: np.ndarray, best_config: Dict, task_type: str, monotonic_indices: List[int],
                   n_splits: int = 5) -> Tuple[List[float], Dict]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=GLOBAL_SEED)
    scores = []
    mono_metrics_list = []
    best_config["hidden_sizes"] = ast.literal_eval(best_config["hidden_sizes"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        g = torch.Generator()
        g.manual_seed(GLOBAL_SEED + fold)

        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).reshape(-1, 1))
        train_loader = DataLoader(train_dataset, batch_size=best_config["batch_size"], shuffle=True, generator=g)
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val).reshape(-1, 1))
        val_loader = DataLoader(val_dataset, batch_size=best_config["batch_size"], generator=g)

        model = create_model(best_config, X.shape[1], task_type, GLOBAL_SEED + fold).to(device)
        optimizer = AdamWScheduleFree(model.parameters(), lr=best_config["lr"], warmup_steps=5)
        _ = train_model(model, optimizer, train_loader, val_loader, best_config, task_type, device)

        val_metric, mono_metrics = evaluate_with_monotonicity(model, optimizer, train_loader, val_loader, task_type,
                                                              device, monotonic_indices)
        scores.append(val_metric)
        mono_metrics_list.append(mono_metrics)

    avg_mono_metrics = {
        'random': np.mean([m['random'] for m in mono_metrics_list]),
        'train': np.mean([m['train'] for m in mono_metrics_list]),
        'val': np.mean([m['val'] for m in mono_metrics_list])
    }

    return scores, avg_mono_metrics


def repeated_train_test(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                        best_config: Dict, task_type: str, monotonic_indices: List[int], n_repeats: int = 5) -> Tuple[
    List[float], Dict]:
    scores = []
    mono_metrics_list = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_config["hidden_sizes"] = ast.literal_eval(best_config["hidden_sizes"])

    for i in range(n_repeats):
        np.random.seed(GLOBAL_SEED + i)
        indices = np.random.permutation(len(X_train))
        X_train, y_train = X_train[indices], y_train[indices]

        g = torch.Generator()
        g.manual_seed(GLOBAL_SEED + i)

        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).reshape(-1, 1))
        train_loader = DataLoader(train_dataset, batch_size=best_config["batch_size"], shuffle=True, generator=g)
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test).reshape(-1, 1))
        test_loader = DataLoader(test_dataset, batch_size=best_config["batch_size"], generator=g)

        model = create_model(best_config, X_train.shape[1], task_type, GLOBAL_SEED + i).to(device)
        optimizer = AdamWScheduleFree(model.parameters(), lr=best_config["lr"])
        _ = train_model(model, train_loader, test_loader, best_config, task_type, device)

        test_metric, mono_metrics = evaluate_with_monotonicity(model, optimizer, train_loader, test_loader, task_type,
                                                               device, monotonic_indices)
        scores.append(test_metric)
        mono_metrics_list.append(mono_metrics)

    avg_mono_metrics = {
        'random': np.mean([m['random'] for m in mono_metrics_list]),
        'train': np.mean([m['train'] for m in mono_metrics_list]),
        'test': np.mean([m['val'] for m in mono_metrics_list])
    }

    return scores, avg_mono_metrics


def process_dataset(data_loader: Callable, sample_size: int = 50000) -> Tuple[List[float], Dict, Dict]:
    print(f"\nProcessing dataset: {data_loader.__name__}")
    X, y, X_test, y_test = data_loader()
    task_type = get_task_type(data_loader)
    monotonic_indices = get_monotonic_indices(data_loader.__name__)
    n_trials = 30
    best_config = optimize_hyperparameters(X, y, task_type, sample_size=sample_size, n_trials=n_trials)

    if data_loader == load_blog_feedback:
        scores, mono_metrics = repeated_train_test(X, y, X_test, y_test, best_config, task_type, monotonic_indices)
    else:
        X = np.vstack((X, X_test))
        y = np.concatenate((y, y_test))
        scores, mono_metrics = cross_validate(X, y, best_config, task_type, monotonic_indices)

    return scores, best_config, mono_metrics


def main():
    set_global_seed(GLOBAL_SEED)

    dataset_loaders = [
        load_abalone, load_auto_mpg, load_blog_feedback, load_boston_housing,
        load_compas, load_era, load_esl, load_heart, load_lev, load_loan, load_swd
    ]

    dataset_loaders = [load_auto_mpg, load_swd]

    sample_size = 30000

    for data_loader in dataset_loaders:
        task_type = get_task_type(data_loader)
        scores, best_config, mono_metrics = process_dataset(data_loader, sample_size)
        metric_name = "RMSE" if task_type == "regression" else "Error Rate"
        print(f"\nResults for {data_loader.__name__}:")
        print(f"{metric_name}: {np.mean(scores):.4f} (±{np.std(scores):.4f})")
        print(f"Best configuration: {best_config}")
        print("Monotonicity violation rates:")
        print(f"  Random data: {mono_metrics['random']:.4f}")
        print(f"  Train data: {mono_metrics['train']:.4f}")
        print(f"  Test data: {mono_metrics['val']:.4f}")

if __name__ == '__main__':
    main()