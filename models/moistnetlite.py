"""
MoistNetLite model architecture.

This module defines the lightweight convolutional neural network used
for vision-based wood-chip moisture classification in Wood-Chip Monitor.

The trained model weights and TensorRT engine are not distributed in
this repository. Deployment uses a TensorRT-optimized representation
of the trained model.
"""

from tensorflow.keras import backend as K
from tensorflow.keras.activations import softmax
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Dropout,
    Dense,
    GlobalAveragePooling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# RandomTranslation moved between TensorFlow/Keras versions.
try:
    from tensorflow.keras.layers import RandomTranslation
except ImportError:
    from tensorflow.keras.layers.experimental.preprocessing import (
        RandomTranslation,
    )


def recall_metric(y_true, y_pred):
    """Compute batch-wise recall."""

    true_positives = K.sum(
        K.round(K.clip(y_true * y_pred, 0, 1))
    )

    possible_positives = K.sum(
        K.round(K.clip(y_true, 0, 1))
    )

    return true_positives / (
        possible_positives + K.epsilon()
    )


def precision_metric(y_true, y_pred):
    """Compute batch-wise precision."""

    true_positives = K.sum(
        K.round(K.clip(y_true * y_pred, 0, 1))
    )

    predicted_positives = K.sum(
        K.round(K.clip(y_pred, 0, 1))
    )

    return true_positives / (
        predicted_positives + K.epsilon()
    )


def f1_metric(y_true, y_pred):
    """Compute batch-wise F1 score."""

    precision = precision_metric(
        y_true,
        y_pred,
    )

    recall = recall_metric(
        y_true,
        y_pred,
    )

    return 2 * (
        (precision * recall)
        / (precision + recall + K.epsilon())
    )


def build_moistnet_lite(
    learning_rate,
    num_filters,
    num_layers,
    dropout_rate,
    dense_layer_size,
    compile_model=True,
):
    """
    Build the MoistNetLite architecture.

    Parameters
    ----------
    learning_rate : float
        Adam optimizer learning rate.

    num_filters : int
        Number of convolution filters used in the standard blocks.

    num_layers : int
        Number of convolution/pooling blocks.

    dropout_rate : float
        Dropout probability applied after each pooling layer.

    dense_layer_size : int
        Number of units in the final hidden dense layer.

    compile_model : bool, optional
        If True, compile the model using the training objective and
        metrics used during development.

    Returns
    -------
    tensorflow.keras.Model
        MoistNetLite model.
    """

    inputs = Input(
        shape=(224, 224, 3),
        name="input_1",
    )

    x = RandomTranslation(
        height_factor=0.1,
        width_factor=0.1,
        fill_mode="reflect",
        name="random_translation",
    )(inputs)

    for layer_index in range(num_layers):

        # Preserve the architecture used during model development.
        filters = (
            512
            if layer_index == 2
            else num_filters
        )

        x = Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            activation="relu",
            name=f"conv2d_{layer_index}_0",
        )(x)

        x = MaxPooling2D(
            pool_size=(2, 2),
            name=f"max_pooling2d_{layer_index}",
        )(x)

        x = Dropout(
            rate=dropout_rate,
            name=f"dropout_{layer_index}",
        )(x)

    x = GlobalAveragePooling2D(
        name="global_average_pooling2d"
    )(x)

    x = Dense(
        units=dense_layer_size,
        activation="relu",
        name="dense",
    )(x)

    outputs = Dense(
        units=3,
        activation=softmax,
        name="classification_head_2",
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="MoistNetLite",
    )

    if compile_model:

        optimizer = Adam(
            learning_rate=learning_rate
        )

        model.compile(
            loss="categorical_crossentropy",
            optimizer=optimizer,
            metrics=[
                "accuracy",
                precision_metric,
                recall_metric,
                f1_metric,
            ],
        )

    return model