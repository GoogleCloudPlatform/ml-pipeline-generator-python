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
"""Executes model training and evaluation."""

import argparse
import json
import logging
import os
import sys

import hypertune
import numpy as np
from sklearn import model_selection

from trainer import inputs
from trainer import model
from trainer import utils


def _parse_arguments(argv):
    """Parses execution arguments and replaces default values.

    Args:
      argv: Input arguments from sys.

    Returns:
      Dictionary of parsed arguments.
    """
    parser = argparse.ArgumentParser()

    # TODO(humichael): Make this into modular template.
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


# TODO(humichael): Evaluate the results.
def _train_and_evaluate(estimator, dataset, model_dir, params):
    """Runs model training and evaluation."""
    x_train, y_train, x_eval, y_eval = dataset
    estimator.fit(x_train, y_train)

    model_path = os.path.join(model_dir, "model.joblib")
    utils.dump_object(estimator, model_path)

    scores = model_selection.cross_val_score(
        estimator, x_eval, y_eval, cv=params.cross_validations)
    metric_path = os.path.join(model_dir, "eval_metrics.joblib")
    utils.dump_object(scores, metric_path)

    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag="score",
        metric_value=np.mean(scores))


def _get_trial_id():
    """Returns the trial id if it exists, else "0"."""
    trial_id = json.loads(
        os.environ.get("TF_CONFIG", "{}")).get("task", {}).get("trial", "")
    return trial_id if trial_id else "1"


def run_experiment(params):
    """Testbed for running model training and evaluation."""
    dataset = inputs.download_data(params.train_path, params.eval_path)
    estimator = model.get_estimator(params)
    trial_id = _get_trial_id()
    model_dir = os.path.join(params.model_dir, trial_id)
    _train_and_evaluate(estimator, dataset, model_dir, params)


def main():
    """Entry point."""
    args = _parse_arguments(sys.argv)
    logging.basicConfig(level="INFO")
    run_experiment(args)


if __name__ == "__main__":
    main()
