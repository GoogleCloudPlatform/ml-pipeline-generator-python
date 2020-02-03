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
"""Executes model training and evaluation."""

import argparse
import logging
import os
import json
import tensorflow as tf

from trainer import model


def _parse_arguments():
    """Parses execution arguments and replaces default values.

    Returns:
      Dictionary of parsed arguments.
    """
    parser = argparse.ArgumentParser()
    {% for arg in args %}
    parser.add_argument(
        "--{{arg.name}}",
        help="{{arg.help}}",
        {% if arg.type|string == "str" %}
        default="{{arg.default}}",
        {% else  %}
        default={{arg.default}},
        {% endif %}
        type={{arg.type}})
    {% endfor %}
    args, _ = parser.parse_known_args()
    return args

def get_type(arg_type):
    if arg_type == "str":
        return str
    elif arg_type == "int":
        return int
    elif arg_type == "float":
        return float


def _get_session_config_from_env_var():
    """Returns a tf.ConfigProto instance that has appropriate device_filters
    set."""

    tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))

    if (tf_config and 'task' in tf_config and 'type' in tf_config['task'] and
            'index' in tf_config['task']):
        # Master should only communicate with itself and ps
        if tf_config['task']['type'] == 'master':
            return tf.ConfigProto(device_filters=['/job:ps', '/job:master'])
        # Worker should only communicate with itself and ps
        elif tf_config['task']['type'] == 'worker':
            return tf.ConfigProto(device_filters=[
                '/job:ps',
                '/job:worker/task:%d' % tf_config['task']['index']
            ])
    return None


def train_and_evaluate(args):
    """Run the training and evaluate using the high level API."""

    def train_input():
        """Input function returning batches from the training
        data set from training.
        """
        return model.input_fn(
            args.train_files,
            batch_size=args.train_batch_size,
            prefetch_buffer_size=args.prefetch_buffer_size)

    def eval_input():
        """Input function returning the entire validation data
        set for evaluation. Shuffling is not required.
        """
        return model.input_fn(
            args.eval_files,
            batch_size=args.eval_batch_size,
            shuffle=False,
            prefetch_buffer_size=args.prefetch_buffer_size)

    train_spec = tf.estimator.TrainSpec(
        train_input, max_steps=args.train_steps)

    exporter = tf.estimator.FinalExporter(
        'exporter', model.SERVING_FUNCTIONS[args.export_format])
    eval_spec = tf.estimator.EvalSpec(
        eval_input,
        steps=args.eval_steps,
        exporters=[exporter],
        name='eval-output')

    run_config = tf.estimator.RunConfig(
        session_config=_get_session_config_from_env_var())
    run_config = run_config.replace(model_dir=args.job_dir)
    print('Model dir %s' % run_config.model_dir)
    estimator = model.build_estimator(
        embedding_size=args.embedding_size,
        # Construct layers sizes with exponential decay
        hidden_units=[
            max(2, int(args.first_layer_size * args.scale_factor**i))
            for i in range(args.num_layers)
        ],
        config=run_config)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    args = _parse_arguments()
    # Set python level verbosity
    logging.basicConfig(level="INFO")
    # Suppress C++ level warnings.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Run the training job
    train_and_evaluate(args)