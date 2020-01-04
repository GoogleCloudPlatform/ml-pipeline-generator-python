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
from pathlib import Path
import sys

import joblib

from trainer import model
from {{model_path}} import get_data


def _train_and_evaluate(estimator, dataset, output_dir):
    """Runs model training and evaluation."""
    x_train, y_train = dataset
    estimator.fit(x_train, y_train)

    model_output_path = os.path.join(output_dir, "{{model_name}}.joblib")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, model_output_path)
    print("Model written to {}".format(model_output_path))


def run_experiment(args):
    """Testbed for running model training and evaluation."""
    dataset = get_data()
    estimator = model.get_estimator(args)
    _train_and_evaluate(estimator, dataset, "{{output_dir}}")


def _parse_args(argv):
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    {{args}}
    return parser.parse_args(argv)


def main():
    """Entry point."""
    args = _parse_args(sys.argv[1:])
    # TODO(humichael): Set log level in args in config
    # logging.basicConfig(level=args.log_level.upper())
    logging.basicConfig(level="INFO")
    run_experiment(args)


if __name__ == "__main__":
    main()
