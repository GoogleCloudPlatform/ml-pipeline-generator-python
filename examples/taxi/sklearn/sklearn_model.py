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
"""Train a simple SVM classifier."""

import argparse
import numpy as np
from sklearn import svm

from examples.preprocess.taxi_preprocess import load_data


def get_model(params):
    """Trains a classifier."""
    classifier = svm.SVC(C=params.C)
    return classifier


def main():
    """Trains a model locally to test get_model()."""
    train_x, train_y, eval_x, eval_y = load_data()
    train_y, eval_y = [np.ravel(x) for x in [train_y, eval_y]]
    params = argparse.Namespace(C=1.0)
    model = get_model(params)
    model.fit(train_x, train_y)
    score = model.score(eval_x, eval_y)
    print(score)


if __name__ == "__main__":
    main()
