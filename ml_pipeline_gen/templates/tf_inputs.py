# Copyright 2019 Google Inc. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Input functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

SCHEMA = {{ schema }}
TARGET = "{{ target }}"


def _decode_csv(line):
    """Takes the string input tensor and returns a dict of rank-2 tensors."""
    columns = tf.decode_csv(line, record_defaults=[0.0] * len(SCHEMA))
    features = dict(zip(SCHEMA, columns))
    for key, _ in six.iteritems(features):
        features[key] = tf.expand_dims(features[key], -1)
    return features


def get_input_fn(file_pattern, shuffle, batch_size, num_epochs=None,
                 data_format="csv"):
    """Returns an input function.

    Two input methods are supported:
      TFRecord on GCS: provide a file_pattern.
      Local CSV: provide features and labels.

    Args:
      file_pattern: pattern of the input files.
      shuffle: boolean for whether to shuffle the data or not (set True for
        training, False for evaluation)
      batch_size: batch size used to read data.
      num_epochs: number of times to iterate over the dataset.
      data_format: format of input data.

    Returns:
      An input_fn.

    Raises:
      RuntimeError: either file_pattern or features and labels were not
        provided.
    """
    def _csv_input_fn():
        """Parses csv input using tf.data."""
        filenames = tf.io.gfile.glob(file_pattern)
        dataset = tf.data.TextLineDataset(filenames).map(
            _decode_csv,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=batch_size * 10)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=10)
        features = dataset.make_one_shot_iterator().get_next()
        return features, features.pop(TARGET)

    data_formats = {
        "csv": _csv_input_fn,
    }
    if data_format in data_formats:
        return data_formats[data_format]
    raise RuntimeError("Invalid arguments")


def get_serving_input_fn(data_format):
    """Returns a serving input function based on the given format.

    Args:
      data_format: format of input data.

    Returns:
      An input fn for serving.

    Raises:
      KeyError: the given data_format is invalid.
    """

    def _csv_serving_input_fn():
        """Build the serving inputs."""
        csv_row = tf.placeholder(shape=[None], dtype=tf.string)
        features = _decode_csv(csv_row)
        return tf.estimator.export.ServingInputReceiver(
            features, {"csv_row": csv_row})

    def _json_serving_input_fn():
        """Build the serving inputs."""
        inputs = {}
        for col in SCHEMA:
            if col != TARGET:
                inputs[col] = tf.placeholder(shape=[None], dtype=float)
        return tf.estimator.export.ServingInputReceiver(inputs, inputs)

    data_formats = {
        "csv": _csv_serving_input_fn,
        "json": _json_serving_input_fn,
    }
    if data_format in data_formats:
        return data_formats[data_format]
    raise KeyError("Invalid arguments")
