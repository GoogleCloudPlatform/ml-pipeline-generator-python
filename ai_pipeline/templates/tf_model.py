# python3
# Copyright 2020 Google Inc. All Rights Reserved.
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
"""ML model definitions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import tensorflow.compat.v1 as tf

from trainer import inputs
from {{ model_path }} import get_model
from {{ model_path }} import get_loss


# pylint: disable=unused-argument
def _model_fn(features, labels, mode, params):
    """Builds an EstimatorSpec.

    Args:
      features: a dict mapping feature names to tensors.
      labels: a tensor of labels.
      mode: a tf.estimator.ModeKey signifying the Estimator mode.
      params: hyperparameters for the model.

    Returns:
      an EstimatorSpec that defines the model to be run by an Estimator.
    """
    schema = [x for x in inputs.SCHEMA if x != inputs.TARGET]
    feature_columns = [tf.feature_column.numeric_column(
        col, shape=(1,), dtype=tf.dtypes.float32) for col in schema]
    input_layer = tf.feature_column.input_layer(features, feature_columns)
    # TODO(humichael): support multiple outputs.
    predictions = get_model(input_layer, params)

    if mode == tf.estimator.ModeKeys.PREDICT:
        prediction_out = {
            "predictions": predictions,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=prediction_out)

    loss = get_loss()(labels, predictions)
    metrics = {}

    {% for metric in metrics%}
    key = "{{ metric }}"
    # TODO(humichael): how to generate this from user?
    # may tie in with multiple outputs. Use logits for loss, preds for eval.
    predictions = tf.round(predictions)
    metric = tf.metrics.{{ metric }}(labels, predictions)
    metrics[key] = metric
    tf.summary.scalar(key, metric[1])
    {% endfor %}

    tf.summary.merge_all()

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    optimizer = tf.train.AdagradOptimizer(learning_rate=params.learning_rate)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    hook = tf.estimator.LoggingTensorHook(
        [input_layer[:5], labels[:5], predictions[:5]], at_end=True)
    return tf.estimator.EstimatorSpec(
        mode, loss=loss, train_op=train_op, training_hooks=[hook])


def _get_trial_id():
    """Returns the trial id if it exists, else "0"."""
    trial_id = json.loads(
        os.environ.get("TF_CONFIG", "{}")).get("task", {}).get("trial", "")
    return trial_id if trial_id else "1"


def get_estimator(params):
    """Returns a tf.Estimator for reconstruction.

    Args:
      params: a dict of hyperparameters for the model.

    Returns:
      A tf.Estimator.
    """
    config = tf.estimator.RunConfig(
        save_checkpoints_steps=params.save_checkpoints_steps,
        keep_checkpoint_max=params.keep_checkpoint_max,
        log_step_count_steps=params.log_step_count_steps)
    trial_id = _get_trial_id()
    model_dir = os.path.join(params.model_dir, trial_id)

    estimator = tf.estimator.Estimator(
        model_fn=_model_fn,
        model_dir=model_dir,
        config=config,
        params=params)
    return estimator


