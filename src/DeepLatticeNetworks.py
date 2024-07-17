import tensorflow as tf
import tensorflow_lattice as tfl
import numpy as np
from typing import List, Tuple
from dataPreprocessing.loaders import *


def create_deep_lattice_network(
        input_dim: int,
        monotonic_indices: List[int],
        num_lattices: int = 10,
        lattice_size: int = 3,
        num_calibration_keypoints: int = 10,
        use_linear_embedding: bool = True,
        embedding_dim: int = 8,
        use_final_calibration: bool = True
) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(input_dim,))

    # First calibration layer
    calibrated = [tfl.layers.PWLCalibration(
        input_keypoints=np.linspace(0, 1, num=num_calibration_keypoints),
        output_min=0.0,
        output_max=lattice_size - 1.0,
        monotonicity='increasing' if i in monotonic_indices else 'none'
    )(inputs[:, i:i + 1]) for i in range(input_dim)]

    calibrated = tf.keras.layers.Concatenate()(calibrated)

    if use_linear_embedding:
        calibrated = tf.keras.layers.Dense(embedding_dim)(calibrated)

    # Lattice layer
    lattice_inputs = tf.split(calibrated, num_or_size_splits=num_lattices, axis=1)
    lattices = [tfl.layers.Lattice(
        lattice_sizes=[lattice_size] * (input_dim // num_lattices),
        monotonicities=['increasing' if (i * (input_dim // num_lattices) + j) in monotonic_indices else 'none'
                        for j in range(input_dim // num_lattices)],
        output_min=0.0,
        output_max=1.0
    )(lattice_input) for i, lattice_input in enumerate(lattice_inputs)]

    combined = tf.keras.layers.Concatenate()(lattices)

    if use_final_calibration:
        output = tfl.layers.PWLCalibration(
            input_keypoints=np.linspace(0, 1, num=num_calibration_keypoints),
            output_min=0.0,
            output_max=1.0
        )(combined)
    else:
        output = tf.keras.layers.Dense(1)(combined)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model


def train_dln(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        monotonic_indices: List[int],
        epochs: int = 200,
        batch_size: int = 32
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Train a Deep Lattice Network model.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training targets.
        X_test (np.ndarray): Testing features.
        y_test (np.ndarray): Testing targets.
        monotonic_indices (List[int]): Indices of monotonic features.
        epochs (int, optional): Number of training epochs. Defaults to 200.
        batch_size (int, optional): Batch size for training. Defaults to 32.

    Returns:
        Tuple[tf.keras.Model, tf.keras.callbacks.History]: Trained model and training history.
    """
    input_dim = X_train.shape[1]

    model = create_deep_lattice_network(input_dim, monotonic_indices)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size
    )

    return model, history


def main():
    """
    Main function to demonstrate the usage of Deep Lattice Network.
    """
    # Use an existing data loader, for example, load_boston_housing
    X_train, y_train, X_test, y_test = load_boston_housing()

    # Specify which features should be monotonic
    # For Boston Housing: 5 (RM) is increasing, 0 (CRIM) is decreasing
    monotonic_indices = [0, 5]

    # Train the model
    model, history = train_dln(X_train, y_train, X_test, y_test, monotonic_indices)

    # Evaluate the model
    test_loss = model.evaluate(X_test, y_test)
    print(f"Test loss: {test_loss}")


if __name__ == '__main__':
    main()
