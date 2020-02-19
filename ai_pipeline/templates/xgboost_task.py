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

from trainer import model


def _get_trial_id():
    """Returns the trial id if it exists, else "0"."""
    trial_id = json.loads(
        os.environ.get("TF_CONFIG", "{}")).get("task", {}).get("trial",
                                                               "")
    return trial_id if trial_id else "1"

# Parse arguments
def _parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--TRAIN_FILE',
        help='Location of the training file',
        default='tmp/train.csv',
        type=str
    )
    parser.add_argument(
        '--TEST_FILE',
        help='Location of the test file',
        default='tmp/test.csv',
        type=str
    )
    parser.add_argument(
        "--output_dir",
        help="Output directory for exporting model and other metadata.",
        default="tmp",
        type=str
    )
    parser.add_argument(
        "--job-dir",
        help="Not used, but needed for AI Platform.",
        default="",
    )
    parser.add_argument(
        '--max_depth',
        help='Maximum depth of the XGBoost tree.',
        default=3,
        type=int
    )
    parser.add_argument(
        '--n_estimators',
        help='Number of estimators to be created.',
        default=2,
        type=int
    )
    parser.add_argument(
        '--booster',
        help='which booster to use: gbtree, gblinear or dart.',
        default='gbtree',
        type=str
    )
    parser.add_argument(
        '--min_child_weight',
        help='Minimum sum of instance weight (hessian) needed in a child',
        default=1,
        type=int
    )
    parser.add_argument(
        '--learning_rate',
        help='Step size shrinkage used in update to prevents overfitting',
        default=0.3,
        type=int
    )
    parser.add_argument(
        '--gamma',
        help='Minimum loss reduction required to make a further partition on a leaf node of the tree',
        default=0,
        type=int
    )
    parser.add_argument(
        '--subsample',
        help='Subsample ratio of the training instances',
        default=1,
        type=int
    )
    parser.add_argument(
        '--colsample_bytree',
        help='subsample ratio of columns when constructing each tree',
        default=1,
        type=int
    )
    parser.add_argument(
        '--reg_alpha',
        help='L1 regularization term on weights. Increasing this value will make model more conservative',
        default=0,
        type=int
    )
    parser.add_argument(
        "--log_level",
        help="Logging level.",
        choices=[
            "DEBUG",
            "ERROR",
            "FATAL",
            "INFO",
            "WARN",
        ],
        default="INFO",
    )
    parser.add_argument(
        "--model_dir",
        help="Not currently used.",
        default="",
    )
    args = parser.parse_args()
    return args


# ------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_arguments(sys.argv)

    {% for arg in args %}
        {% if arg.type|string == "str" %}
    args.{{arg.name}} = "{{arg.value}}"
    {% else %}
    args.{{arg.name}} = {{arg.value}}
        {% endif %}
    {% endfor %}
    logging.basicConfig(level=args.log_level.upper())

    # Append trial_id to path
    trial_id = _get_trial_id()
    model_dir = os.path.join("{{model_dir}}", trial_id)
    print('Model dir %s' % model_dir)

    # Run the training job
    model.train_and_evaluate(args, model_dir)
