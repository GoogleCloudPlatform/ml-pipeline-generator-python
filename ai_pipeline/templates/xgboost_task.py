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
import os
import sys
import json
import logging
import hypertune

from sklearn import metrics
from sklearn import preprocessing

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


def _get_trial_id():
    """Returns the trial id if it exists, else "0"."""
    trial_id = json.loads(
        os.environ.get("TF_CONFIG", "{}")).get("task", {}).get("trial",
                                                               "")
    return trial_id if trial_id else "1"


def _train_and_evaluate(estimator, dataset, model_dir):
    """Runs model training and evaluation."""
    x_train, y_train, x_eval, y_eval = dataset
    estimator.fit(x_train, y_train)
    logging.info("Completed training XGBOOST model")

    bst = estimator.get_booster()
    bst_filename = 'model.bst'
    bst.save_model(bst_filename)
    model_output_path = os.path.join(model_dir, bst_filename)
    utils.upload_blob(model_output_path.split("/")[2], bst_filename,
                      "/".join(model_output_path.split("/")[3:]))
    logging.info("Successfully uploaded file to GCS at location %s",
                 model_dir)
    y_pred = estimator.predict(x_eval)

    # Binarize multiclass labels
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_eval)
    y_test = lb.transform(y_eval)
    y_pred = lb.transform(y_pred)

    score = metrics.roc_auc_score(y_test, y_pred, average='macro')
    logging.info("AUC Score: %s", str(score))

    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='roc_auc',
        metric_value=score,
        global_step=1000
    )


def run_experiment(params):
    """Testbed for running model training and evaluation."""
    dataset = inputs.download_data(params.train_path, params.eval_path)
    estimator = model.get_estimator(params)
    trial_id = _get_trial_id()
    model_dir = os.path.join(params.model_dir, trial_id)
    _train_and_evaluate(estimator, dataset, model_dir)


def main():
    """Entry point."""
    args = _parse_arguments(sys.argv)
    logging.basicConfig(level="INFO")
    run_experiment(args)


if __name__ == "__main__":
    main()
