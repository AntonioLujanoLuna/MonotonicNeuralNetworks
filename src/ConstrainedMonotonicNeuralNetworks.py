import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from airt.keras.layers import MonoDense
from typing import List, Tuple, Union, Optional


class ConstrainedMonotonicNetwork:
    """
    A class to create and manage Constrained Monotonic Neural Networks.

    This class provides a high-level interface for building, training, and using
    neural networks with monotonicity constraints on specific input features.

    Attributes:
        input_shape (Tuple[int, ...]): Shape of the input data.
        hidden_layers (List[int]): Number of units in each hidden layer.
        output_shape (int): Number of units in the output layer.
        monotonicity_indicator (List[int]): Monotonicity constraints for input features.
        activation (str): Activation function for hidden layers.
        learning_rate (float): Initial learning rate for the optimizer.
        decay_steps (int): Number of steps for learning rate decay.
        decay_rate (float): Rate of learning rate decay.
        model (tf.keras.Model): The underlying Keras model.
    """

    def __init__(
            self,
            input_shape: Tuple[int, ...],
            hidden_layers: List[int],
            output_shape: int = 1,
            monotonicity_indicator: Optional[List[int]] = None,
            activation: str = 'elu',
            learning_rate: float = 0.01,
            decay_steps: int = 10000,
            decay_rate: float = 0.9
    ):
        """
        Initialize the ConstrainedMonotonicNetwork.

        Args:
            input_shape (Tuple[int, ...]): Shape of the input data.
            hidden_layers (List[int]): Number of units in each hidden layer.
            output_shape (int): Number of units in the output layer.
            monotonicity_indicator (Optional[List[int]]): Monotonicity constraints for input features.
                Use 1 for increasing, -1 for decreasing, and 0 for unrestricted.
            activation (str): Activation function for hidden layers.
            learning_rate (float): Initial learning rate for the optimizer.
            decay_steps (int): Number of steps for learning rate decay.
            decay_rate (float): Rate of learning rate decay.
        """
        self.input_shape = input_shape
        self.hidden_layers = hidden_layers
        self.output_shape = output_shape
        self.monotonicity_indicator = monotonicity_indicator or [1] * input_shape[0]
        self.activation = activation
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.model = self._build_model()

    def _build_model(self) -> tf.keras.Model:
        """
        Build the Keras model with monotonicity constraints.

        Returns:
            tf.keras.Model: The constructed Keras model.
        """
        model = Sequential()
        model.add(Input(shape=self.input_shape))

        # First layer with monotonicity constraints
        model.add(MonoDense(
            self.hidden_layers[0],
            activation=self.activation,
            monotonicity_indicator=self.monotonicity_indicator
        ))

        # Additional hidden layers
        for units in self.hidden_layers[1:]:
            model.add(MonoDense(units, activation=self.activation))

        # Output layer
        model.add(MonoDense(self.output_shape))

        return model

    def compile(self, loss: str = 'mse', metrics: Optional[List[str]] = None) -> None:
        """
        Compile the model with the specified loss function and metrics.

        Args:
            loss (str): Loss function to use for training.
            metrics (Optional[List[str]]): List of metrics to track during training.
        """
        lr_schedule = ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
        )
        optimizer = Adam(learning_rate=lr_schedule)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(
            self,
            x_train: Union[np.ndarray, tf.Tensor],
            y_train: Union[np.ndarray, tf.Tensor],
            batch_size: int = 32,
            epochs: int = 10,
            validation_data: Optional[Tuple[Union[np.ndarray, tf.Tensor], Union[np.ndarray, tf.Tensor]]] = None,
            **kwargs
    ) -> tf.keras.callbacks.History:
        """
        Train the model on the provided data.

        Args:
            x_train (Union[np.ndarray, tf.Tensor]): Training input data.
            y_train (Union[np.ndarray, tf.Tensor]): Training target data.
            batch_size (int): Number of samples per gradient update.
            epochs (int): Number of epochs to train the model.
            validation_data (Optional[Tuple[Union[np.ndarray, tf.Tensor], Union[np.ndarray, tf.Tensor]]]):
                Data on which to evaluate the loss and any model metrics at the end of each epoch.
            **kwargs: Additional arguments to be passed to the fit method of the model.

        Returns:
            tf.keras.callbacks.History: A History object containing training loss values and metrics values.
        """
        return self.model.fit(
            x=x_train,
            y=y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            **kwargs
        )

    def predict(self, x: Union[np.ndarray, tf.Tensor]) -> np.ndarray:
        """
        Generate predictions for the input samples.

        Args:
            x (Union[np.ndarray, tf.Tensor]): Input samples.

        Returns:
            np.ndarray: Model predictions.
        """
        return self.model.predict(x)

    def summary(self) -> None:
        """
        Print a string summary of the network.
        """
        self.model.summary()

    def save(self, filepath: str) -> None:
        """
        Save the model to a file.

        Args:
            filepath (str): Path to save the model file.
        """
        self.model.save(filepath)

    @classmethod
    def load(cls, filepath: str) -> 'ConstrainedMonotonicNetwork':
        """
        Load a saved model from a file.

        Args:
            filepath (str): Path to the saved model file.

        Returns:
            ConstrainedMonotonicNetwork: A new instance with the loaded model.
        """
        loaded_model = tf.keras.models.load_model(filepath, custom_objects={'MonoDense': MonoDense})
        instance = cls(input_shape=loaded_model.input_shape[1:])
        instance.model = loaded_model
        return instance

# Example usage
"""
import numpy as np

# Create a constrained monotonic network
cmn = ConstrainedMonotonicNetwork(
    input_shape=(3,),
    hidden_layers=[128, 128],
    monotonicity_indicator=[1, 0, -1],
    activation='elu'
)

# Compile the model
cmn.compile(loss='mse', metrics=['mae'])

# Display model summary
cmn.summary()

# Generate some dummy data for illustration
x_train = np.random.rand(1000, 3)
y_train = np.random.rand(1000, 1)
x_val = np.random.rand(200, 3)
y_val = np.random.rand(200, 1)

# Train the model
history = cmn.fit(
    x_train, y_train,
    batch_size=32,
    epochs=10,
    validation_data=(x_val, y_val)
)

# Make predictions
x_test = np.random.rand(100, 3)
predictions = cmn.predict(x_test)

# Save the model
cmn.save('constrained_monotonic_model.h5')

# Load the model
loaded_cmn = ConstrainedMonotonicNetwork.load('constrained_monotonic_model.h5')
"""