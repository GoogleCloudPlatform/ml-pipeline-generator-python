# python3
# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Train a simple TF classifier for census dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import tensorflow.compat.v1 as tf

from examples.preprocess.census_preprocess import load_data


def get_model(inputs, params):
    """Trains a classifier on iris data."""
    dense = tf.keras.layers.Dense
    nn = dense(params.first_layer_size, activation="relu",
               kernel_initializer="uniform")(inputs)
    for i in reversed(range(1, params.num_layers)):
        layer_size = int(params.first_layer_size * (i / params.num_layers))
        nn = dense(max(1, layer_size), activation="relu")(nn)
    logits = dense(1, activation="sigmoid")(nn)

    return logits


# TODO(humichael): create get_predicition and get_evaluation instead.
def get_loss():
    """The loss function to use."""
    return tf.losses.sigmoid_cross_entropy


def main():
    """Trains a model locally to test get_model() and get_loss()."""
    train_x, train_y, _, _ = load_data()
    input_layer = tf.keras.layers.Input(shape=(train_x.shape[1],))
    params = argparse.Namespace(first_layer_size=50, num_layers=5)
    predictions = get_model(input_layer, params)
    model = tf.keras.models.Model(inputs=input_layer, outputs=predictions)
    model.compile(optimizer="adam", loss=get_loss(),
                  metrics=["accuracy"])
    model.fit(train_x, train_y, epochs=1)


if __name__ == "__main__":
    main()
