import ast
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, accuracy_score
from typing import Callable, Tuple, List, Dict
import optuna
from schedulefree import AdamWScheduleFree
from MLP import StandardMLP
from dataPreprocessing.loaders import (load_abalone, load_auto_mpg, load_blog_feedback, load_boston_housing,
                                       load_compas, load_era, load_esl, load_heart, load_lev, load_loan, load_swd)

def get_task_type(data_loader: Callable) -> str:
    regression_tasks = [load_abalone, load_auto_mpg, load_blog_feedback, load_boston_housing,
                        load_era, load_esl, load_lev, load_swd]
    return "regression" if data_loader in regression_tasks else "classification"

def create_model(config: Dict, input_size: int, task_type: str) -> StandardMLP:
    output_activation = nn.Identity() if task_type == "regression" else nn.Sigmoid()
    return StandardMLP(
        input_size=input_size,
        hidden_sizes=config["hidden_sizes"],
        output_size=1,
        activation=nn.ReLU(),
        output_activation=output_activation,
    )

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                config: Dict, task_type: str, device: torch.device) -> float:
    criterion = nn.MSELoss() if task_type == "regression" else nn.BCELoss()
    optimizer = AdamWScheduleFree(model.parameters(), lr=config["lr"])

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(config["epochs"]):
        model.train()
        optimizer.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # Validation phase
        val_loss = evaluate_model(model, optimizer, val_loader, task_type, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load the best model before returning
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

def objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, task_type: str) -> float:
    config = {
        "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
        "hidden_sizes": ast.literal_eval(trial.suggest_categorical("hidden_sizes",
                                                                   ["[8, 8]", "[8, 16]", "[16, 8]", "[16, 16]",
                                                                    "[16, 32]", "[32, 32]", "[32, 64]", "[64, 64]"])),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        "epochs": 100,  # Ensure epochs is part of the configuration
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y).reshape(-1, 1))
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val).reshape(-1, 1))
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

    model = create_model(config, X.shape[1], task_type).to(device)
    val_metric = train_model(model, train_loader, val_loader, config, task_type, device)

    return val_metric

def optimize_hyperparameters(X: np.ndarray, y: np.ndarray, task_type: str, n_trials: int = 30) -> Dict:
    # Split data into train and validation sets for hyperparameter tuning
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, task_type), n_trials=n_trials, n_jobs=-1)
    best_params = study.best_params
    best_params["epochs"] = 100  # Ensure epochs is included in the best configuration
    return best_params

def cross_validate(X: np.ndarray, y: np.ndarray, best_config: Dict, task_type: str, n_splits: int = 5) -> List[float]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    best_config["hidden_sizes"] = ast.literal_eval(best_config["hidden_sizes"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).reshape(-1, 1))
        train_loader = DataLoader(train_dataset, batch_size=best_config["batch_size"], shuffle=True)
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val).reshape(-1, 1))
        val_loader = DataLoader(val_dataset, batch_size=best_config["batch_size"])

        model = create_model(best_config, X.shape[1], task_type).to(device)
        val_metric = train_model(model, train_loader, val_loader, best_config, task_type, device)
        scores.append(val_metric)

    return scores

def repeated_train_test(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                        best_config: Dict, task_type: str, n_repeats: int = 5) -> List[float]:
    scores = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_config["hidden_sizes"] = ast.literal_eval(best_config["hidden_sizes"])

    for _ in range(n_repeats):
        indices = np.random.permutation(len(X_train))
        X_train, y_train = X_train[indices], y_train[indices]
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).reshape(-1, 1))
        train_loader = DataLoader(train_dataset, batch_size=best_config["batch_size"], shuffle=True)
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test).reshape(-1, 1))
        test_loader = DataLoader(test_dataset, batch_size=best_config["batch_size"])

        model = create_model(best_config, X_train.shape[1], task_type).to(device)
        test_metric = train_model(model, train_loader, test_loader, best_config, task_type, device)
        scores.append(test_metric)

    return scores

def process_dataset(data_loader: Callable) -> Tuple[List[float], Dict]:
    print(f"\nProcessing dataset: {data_loader.__name__}")
    X, y, X_test, y_test = data_loader()
    task_type = get_task_type(data_loader)

    # Hyperparameter optimization
    best_config = optimize_hyperparameters(X, y, task_type)

    # Evaluation
    if data_loader == load_blog_feedback:
        # Special case for blog feedback dataset
        scores = repeated_train_test(X, y, X_test, y_test, best_config, task_type)
    else:
        # stack the training and test data for cross-validation
        X = np.vstack((X, X_test))
        y = np.concatenate((y, y_test))
        scores = cross_validate(X, y, best_config, task_type)

    return scores, best_config

def main():
    dataset_loaders = [
        load_abalone, load_auto_mpg, load_blog_feedback, load_boston_housing,
        load_compas, load_era, load_esl, load_heart, load_lev, load_loan, load_swd
    ]

    for data_loader in dataset_loaders:
        scores, best_config = process_dataset(data_loader)
        task_type = get_task_type(data_loader)
        metric_name = "RMSE" if task_type == "regression" else "Error Rate"
        print(f"\nResults for {data_loader.__name__}:")
        print(f"{metric_name}: {np.mean(scores):.4f} (±{np.std(scores):.4f})")
        print(f"Best configuration: {best_config}")

if __name__ == '__main__':
    main()
