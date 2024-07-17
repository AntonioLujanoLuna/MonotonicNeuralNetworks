import torch
import torch.nn as nn
from pmlayer.torch.layers import HLattice
from typing import List, Optional, Union, Literal
from StandardMLP import StandardMLP  # Assuming StandardMLP is in a file named StandardMLP.py


class HLLNetwork(nn.Module):
    """
    Hybrid Lattice Layer (HLL) Network implementation using pmlayer and StandardMLP.

    This class provides a flexible HLL network with customizable lattice sizes,
    increasing dimensions, and MLP architecture.

    Attributes:
        dim (int): Total number of input dimensions.
        lattice_sizes (List[int]): Sizes of each lattice dimension.
        increasing (List[int]): Indices of input dimensions that should be increasing.
        mlp_neurons (List[int]): Sizes of hidden layers in the MLP.
        activation (nn.Module): Activation function for the MLP.
        dropout_rate (float): Dropout rate for the MLP.
        init_method (str): Weight initialization method for the MLP.
        model (HLattice): The underlying HLattice model.
    """

    def __init__(
            self,
            dim: int,
            lattice_sizes: List[int],
            increasing: List[int],
            mlp_neurons: List[int],
            activation: nn.Module = nn.ReLU(),
            dropout_rate: float = 0.0,
            init_method: Literal[
                'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'truncated_normal'] = 'xavier_uniform'
    ):
        """
        Initialize the HLLNetwork.

        Args:
            dim (int): Total number of input dimensions.
            lattice_sizes (List[int]): Sizes of each lattice dimension.
            increasing (List[int]): Indices of input dimensions that should be increasing.
            mlp_neurons (List[int]): Sizes of hidden layers in the MLP.
            activation (nn.Module): Activation function for the MLP.
            dropout_rate (float): Dropout rate for the MLP.
            init_method (str): Weight initialization method for the MLP.
        """
        super(HLLNetwork, self).__init__()
        self.dim = dim
        self.lattice_sizes = lattice_sizes
        self.increasing = increasing
        self.mlp_neurons = mlp_neurons
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.init_method = init_method

        self.model = self._build_model()

    def _build_model(self) -> HLattice:
        """
        Build the underlying HLattice model.

        Returns:
            HLattice: The constructed HLattice model.
        """
        input_len = self.dim - len(self.increasing)
        output_len = torch.prod(torch.tensor(self.lattice_sizes)).item()

        ann = StandardMLP(
            input_size=input_len,
            hidden_sizes=self.mlp_neurons,
            output_size=output_len,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            init_method=self.init_method
        )

        return HLattice(self.dim, torch.tensor(self.lattice_sizes), self.increasing, ann)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        return self.model(x)

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
        Train the HLL model.

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
        optimizer = torch.optim.Rprop(self.parameters(), lr=learning_rate, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
        criterion = nn.MSELoss()
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_state_dict = None

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

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_state_dict = self.state_dict()
                else:
                    epochs_no_improve += 1

                if early_stopping and epochs_no_improve == patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    self.load_state_dict(best_state_dict)
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

# Example usage
"""
dim = 10
lattice_sizes = [3, 3, 3, 3]  # Example lattice sizes
increasing = [0, 1]  # Example increasing dimensions
mlp_neurons = [64, 32]  # Example MLP architecture

model = HLLNetwork(
    dim=dim,
    lattice_sizes=lattice_sizes,
    increasing=increasing,
    mlp_neurons=mlp_neurons,
    activation=nn.ReLU(),
    dropout_rate=0.1,
    init_method='xavier_uniform'
)

# Assuming you have your data as torch tensors
history = model.fit(x_train, y_train, x_val, y_val, epochs=1000, learning_rate=0.01)

# Make predictions
predictions = model.predict(x_test)
"""