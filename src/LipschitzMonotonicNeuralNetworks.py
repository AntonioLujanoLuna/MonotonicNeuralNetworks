import torch
import torch.nn as nn
import monotonicnetworks as lmn
from typing import List, Optional


class LMNNetwork(nn.Module):
    """
    Lipschitz Monotonic Neural Network (LMNN) implementation.

    This class provides a flexible LMNN with customizable layer sizes, monotonicity constraints,
    and Lipschitz constant.

    Attributes:
        input_size (int): Size of the input layer.
        hidden_sizes (List[int]): Sizes of the hidden layers.
        output_size (int): Size of the output layer.
        monotone_constraints (Optional[List[int]]): List of monotonicity constraints for each input feature.
        lipschitz_constant (float): Lipschitz constant for the network.
        sigma (float): Sigma value for the SigmaNet wrapper.
        model (nn.Module): The underlying neural network model.
        wrapped_model (lmn.SigmaNet): The model wrapped with SigmaNet for monotonicity constraints.
    """

    def __init__(
            self,
            input_size: int,
            hidden_sizes: List[int],
            output_size: int = 1,
            monotone_constraints: Optional[List[int]] = None,
            lipschitz_constant: float = 1.0,
            sigma: float = 1.0
    ):
        """
        Initialize the LMNNNetwork.

        Args:
            input_size (int): Size of the input layer.
            hidden_sizes (List[int]): List of sizes for the hidden layers.
            output_size (int): Size of the output layer (default is 1).
            monotone_constraints (Optional[List[int]]): List of monotonicity constraints for each input feature.
                Use 1 for increasing, -1 for decreasing, and 0 for unrestricted. Default is None (all unrestricted).
            lipschitz_constant (float): Lipschitz constant for the network (default is 1.0).
            sigma (float): Sigma value for the SigmaNet wrapper (default is 1.0).
        """
        super(LMNNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.monotone_constraints = monotone_constraints
        self.lipschitz_constant = lipschitz_constant
        self.sigma = sigma

        self.model = self._build_model()
        self.wrapped_model = lmn.MonotonicWrapper(
            self.model,
            lipschitz_const=self.lipschitz_constant,
            monotonic_constraints=self.monotone_constraints
        )

    def _build_model(self) -> nn.Sequential:
        """
        Build the underlying neural network model.

        Returns:
            nn.Sequential: The constructed neural network model.
        """
        layers = []
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]

        for i in range(len(layer_sizes) - 1):
            if i == 0:
                layers.append(lmn.direct_norm(nn.Linear(layer_sizes[i], layer_sizes[i + 1]), kind="one-inf"))
            else:
                layers.append(lmn.direct_norm(nn.Linear(layer_sizes[i], layer_sizes[i + 1]), kind="inf"))

            if i < len(layer_sizes) - 2:  # Don't add activation after the last layer
                layers.append(lmn.GroupSort(layer_sizes[i + 1] // 2))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        return self.wrapped_model(x)

    def fit(
            self,
            x_train: torch.Tensor,
            y_train: torch.Tensor,
            x_val: Optional[torch.Tensor] = None,
            y_val: Optional[torch.Tensor] = None,
            epochs: int = 1000,
            learning_rate: float = 0.01,
            batch_size: int = 32,
            early_stopping: bool = True,
            patience: int = 10
    ) -> dict:
        """
        Train the LMNN model.

        Args:
            x_train (torch.Tensor): Training input data.
            y_train (torch.Tensor): Training target data.
            x_val (Optional[torch.Tensor]): Validation input data.
            y_val (Optional[torch.Tensor]): Validation target data.
            epochs (int): Maximum number of training epochs.
            learning_rate (float): Learning rate for the optimizer.
            batch_size (int): Batch size for training.
            early_stopping (bool): Whether to use early stopping.
            patience (int): Number of epochs with no improvement after which training will be stopped.

        Returns:
            dict: A dictionary containing training history (loss and validation loss if applicable).
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            for i in range(0, len(x_train), batch_size):
                batch_x = x_train[i:i + batch_size]
                batch_y = y_train[i:i + batch_size]

                optimizer.zero_grad()
                outputs = self(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= (len(x_train) // batch_size)
            history['train_loss'].append(train_loss)

            if x_val is not None and y_val is not None:
                self.eval()
                with torch.no_grad():
                    val_outputs = self(x_val)
                    val_loss = criterion(val_outputs, y_val).item()
                history['val_loss'].append(val_loss)

                if early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve == patience:
                            print(f"Early stopping triggered at epoch {epoch + 1}")
                            break

            print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}", end="")
            if x_val is not None:
                print(f", Validation Loss: {val_loss:.4f}")
            else:
                print()

        return history

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions using the trained model.

        Args:
            x (torch.Tensor): Input data for predictions.

        Returns:
            torch.Tensor: Predicted output.
        """
        self.eval()
        with torch.no_grad():
            return self(x)

    def count_parameters(self) -> int:
        """
        Count the number of trainable parameters in the network.

        Returns:
            int: The total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)