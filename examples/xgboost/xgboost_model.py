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
"""Train a simple XGBoost classifier for Iris dataset."""

from sklearn import datasets
from xgboost import XGBClassifier


def get_data():
    iris = datasets.load_iris()
    return [iris.data, iris.target]


def get_model(args={}):
    """Trains a classifier on iris data."""
    classifier = XGBClassifier()
    return classifier


if __name__ == "__main__":
    data, target = get_data()
    model = get_model()
    model.fit(data, target)
