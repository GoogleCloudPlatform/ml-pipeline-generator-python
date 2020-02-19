# python3
# Copyright 2020 Google Inc. All Rights Reserved.
#
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
"""Main script to train the model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow.compat.v1 as tf

from trainer import inputs
from trainer import model


def _parse_arguments(argv):
    """Parses execution arguments and replaces default values.

    Args:
      argv: Input arguments from sys.

    Returns:
      Dictionary of parsed arguments.
    """
    parser = argparse.ArgumentParser()

    {% for name, arg in input_args.items() %}
    parser.add_argument(
        "--{{name}}",
        help="{{arg.help}}",
        type={{arg.type}},
        {% if arg.type == "str" and "default" in arg %}
        default="{{arg.default}}",
        {% elif "default" in arg %}
        default={{arg.default}},
        {% endif %}
    )
    {% endfor %}

    args, _ = parser.parse_known_args(args=argv[1:])
    return args


def run_training(params):
    """Initializes the estimator and runs train_and_evaluate."""
    estimator = model.get_estimator(params)
    train_input_fn = inputs.get_input_fn(
        params.train_path,
        shuffle=True,
        batch_size=params.batch_size,
        num_epochs=params.num_epochs,
    )
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=params.max_steps,
    )
    eval_input_fn = inputs.get_input_fn(
        params.eval_path,
        shuffle=False,
        batch_size=params.batch_size,
    )
    exporter = tf.estimator.BestExporter(
        "export", inputs.get_serving_input_fn(params.export_format),
        exports_to_keep=1)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn,
        throttle_secs=1,
        steps=params.eval_steps,
        start_delay_secs=1,
        exporters=[exporter],
    )
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def main():
    """Trains a model."""
    params = _parse_arguments(sys.argv)
    tf.logging.set_verbosity(tf.logging.INFO)
    run_training(params)


if __name__ == "__main__":
    main()
