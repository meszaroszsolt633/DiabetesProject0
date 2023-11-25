import torch
import torch.nn as nn
import math
import numpy as np
from tqdm import tqdm
from torch.nn.functional import mse_loss, l1_loss

import pandas as pd
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt

from functions import loadeverycleanedxml
from xml_read import load
from defines import *
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers

from modelMealClassificationCNN import dataPrepare

#train, _= load(TRAIN2_544_PATH)
#test,_= load(TEST2_544_PATH)
train,test=loadeverycleanedxml()
x_train, y_train, x_test, y_test = dataPrepare(train, test, 3, 3,15,15,False)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def recall_m(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision

def fbeta_m(y_true, y_pred, beta=2):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    beta_squared = beta ** 2
    fbeta = (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall + tf.keras.backend.epsilon())
    return fbeta
n_classes=1
def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)

input_shape = x_train.shape[1:]

model = build_model(
    input_shape,
    head_size=256,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=[128],
    mlp_dropout=0.4,
    dropout=0.25,
)

model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["accuracy",recall_m,
        precision_m,
        fbeta_m,]
)
model.summary()

callbacks = [keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True)]

history=model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=2000,
    batch_size=64,
    callbacks=callbacks,
)

prediction = model.predict(x_test)
    # Prediction and actual data plot
plt.figure(figsize=(20, 6))
plt.plot(prediction[0:1440 * 3], label='prediction')
plt.plot(y_test[0:1440 * 3], label='test_data')
plt.legend()
plt.show()
# Loss and validation loss plot
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Training VS Validation loss')
plt.xlabel('No. of Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.evaluate(x_test, y_test, verbose=1)

