import torch
import torch.nn as nn
import torch.autograd as autograd
from typing import List, Literal, Union
from src.MLP import StandardMLP
from src.WeightsConstrainedMLP import WeightsConstrainedMLP

class PartialMonotonicNetwork(nn.Module):
    """
    Partial Monotonic Network as described in the article.

    This network combines monotonic and non-monotonic features processing,
    allowing for partial monotonicity constraints in the model.

    Attributes:
        monotonic_indices (List[int]): Indices of monotonic features.
        non_monotonic_indices (List[int]): Indices of non-monotonic features.
        mono_network (WeightsConstrainedMLP): Network for monotonic features.
        non_mono_network (StandardMLP): Network for non-monotonic features.
        combined_network (WeightsConstrainedMLP): Network for combining features.
        p (float): Hyperparameter for balancing empirical and monotonic losses.
        s (nn.Parameter): Dynamic scale parameter for monotonic loss.
    """

    def __init__(
        self,
        input_size: int,
        monotonic_indices: List[int],
        mono_hidden_sizes: List[int],
        non_mono_hidden_sizes: List[int],
        combined_hidden_sizes: List[int],
        output_size: int = 1,
        activation: Union[str, nn.Module] = 'leaky_relu',
        output_activation: Union[str, nn.Module] = 'identity',
        dropout_rate: float = 0.0,
        init_method: Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'truncated_normal'] = 'xavier_uniform',
        transform: Literal['exp', 'explin', 'sqr'] = 'exp',
        p: float = 0.1
    ):
        """
        Initialize the PartialMonotonicNetwork.

        Args:
            input_size (int): Total number of input features.
            monotonic_indices (List[int]): Indices of monotonic features.
            mono_hidden_sizes (List[int]): Hidden layer sizes for monotonic network.
            non_mono_hidden_sizes (List[int]): Hidden layer sizes for non-monotonic network.
            combined_hidden_sizes (List[int]): Hidden layer sizes for combined network.
            output_size (int): Number of output units.
            activation (Union[str, nn.Module]): Activation function for hidden layers.
            output_activation (Union[str, nn.Module]): Activation function for output layer.
            dropout_rate (float): Dropout rate for all networks.
            init_method (str): Weight initialization method.
            transform (str): Weight transformation method for monotonic networks.
            p (float): Hyperparameter for balancing empirical and monotonic losses.
        """
        super(PartialMonotonicNetwork, self).__init__()
        self.monotonic_indices = monotonic_indices
        self.non_monotonic_indices = [i for i in range(input_size) if i not in monotonic_indices]
        self.p = p
        self.input_size = input_size

        # Convert string activations to nn.Module
        activation = self._get_activation(activation)
        output_activation = self._get_activation(output_activation)

        # Monotonic features network
        self.mono_network = WeightsConstrainedMLP(
            input_size=len(monotonic_indices),
            hidden_sizes=mono_hidden_sizes,
            output_size=mono_hidden_sizes[-1],
            activation=activation,
            output_activation=activation,
            dropout_rate=dropout_rate,
            init_method=init_method,
            transform=transform
        )

        # Non-monotonic features network
        self.non_mono_network = StandardMLP(
            input_size=len(self.non_monotonic_indices),
            hidden_sizes=non_mono_hidden_sizes,
            output_size=non_mono_hidden_sizes[-1],
            activation=activation,
            output_activation=activation,
            dropout_rate=dropout_rate,
            init_method=init_method
        )

        # Combined network
        self.combined_network = WeightsConstrainedMLP(
            input_size=mono_hidden_sizes[-1] + non_mono_hidden_sizes[-1],
            hidden_sizes=combined_hidden_sizes,
            output_size=output_size,
            activation=activation,
            output_activation=output_activation,
            dropout_rate=dropout_rate,
            init_method=init_method,
            transform=transform
        )

        self.s = nn.Parameter(torch.tensor(1.0))
        self.mono_loss_history = []
        self.nn_loss_history = []

    @staticmethod
    def _get_activation(activation: Union[str, nn.Module]) -> nn.Module:
        """
        Convert string activation to nn.Module.

        Args:
            activation (Union[str, nn.Module]): Activation function or its name.

        Returns:
            nn.Module: Activation function as nn.Module.
        """
        if isinstance(activation, str):
            return {
                'relu': nn.ReLU(),
                'leaky_relu': nn.LeakyReLU(),
                'tanh': nn.Tanh(),
                'sigmoid': nn.Sigmoid(),
                'identity': nn.Identity(),
            }.get(activation.lower(), nn.ReLU())
        return activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        mono_features = x[:, self.monotonic_indices]
        non_mono_features = x[:, self.non_monotonic_indices]

        mono_output = self.mono_network(mono_features)
        non_mono_output = self.non_mono_network(non_mono_features)

        combined_features = torch.cat((mono_output, non_mono_output), dim=1)
        output = self.combined_network(combined_features)

        return output

    def compute_monotonic_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the monotonicity enforcing loss (PWL).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Monotonicity loss.
        """
        monotonic_inputs = x[:, self.monotonic_indices]
        monotonic_inputs.requires_grad_(True)
        mono_output = self.mono_network(monotonic_inputs)
        non_mono_output = self.non_mono_network(x[:, self.non_monotonic_indices])
        combined_features = torch.cat((mono_output, non_mono_output), dim=1)
        output = self.combined_network(combined_features)
        monotonic_gradients = autograd.grad(outputs=output.sum(), inputs=monotonic_inputs,
                                            create_graph=True, retain_graph=True)[0]
        pwl = torch.sum(torch.relu(-monotonic_gradients))
        return pwl

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor, loss_fn: nn.Module) -> torch.Tensor:
        """
        Compute the total loss including both empirical and monotonicity loss.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
            y (torch.Tensor): Target tensor of shape (batch_size, output_size).
            loss_fn (nn.Module): Loss function for empirical loss.

        Returns:
            torch.Tensor: Total loss.
        """
        y_pred = self.forward(x)
        nn_loss = loss_fn(y_pred, y)
        mono_loss = self.compute_monotonic_loss(x)

        self.mono_loss_history.append(mono_loss.item())
        self.nn_loss_history.append(nn_loss.item())

        if len(self.mono_loss_history) > 1:
            m_mono = sum(self.mono_loss_history) / len(self.mono_loss_history)
            m_nn = sum(self.nn_loss_history) / len(self.nn_loss_history)
            r = m_nn / m_mono if m_mono != 0 else 1.0
            s = 10 ** int(torch.log10(torch.tensor(r)))
        else:
            s = 1.0

        total_loss = (1 - self.p) * nn_loss + self.p * s * mono_loss
        return total_loss

    def count_parameters(self) -> int:
        """
        Count the number of trainable parameters in the network.

        Returns:
            int: The total number of trainable parameters.
        """
        return (
            self.mono_network.count_parameters() +
            self.non_mono_network.count_parameters() +
            self.combined_network.count_parameters()
        )