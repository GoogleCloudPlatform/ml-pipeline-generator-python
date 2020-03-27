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
"""Input functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

SCHEMA = {{ schema }}
TARGET = "{{ target }}"


def download_data(train_path, eval_path):
    """Downloads train and eval datasets from GCP.

    Args:
        train_path: GCS path to training data.
        eval_path: GCS path to evaluation data.

    Returns:
        train_x: dataframe of training features.
        train_y: dataframe of training labels.
        eval_x: dataframe of eval features.
        eval_y: dataframe of eval labels.
    """

    train_df = pd.read_csv(train_path, names=SCHEMA)
    eval_df = pd.read_csv(eval_path, names=SCHEMA)
    train_x, train_y = train_df.drop(TARGET, axis=1), train_df[TARGET]
    eval_x, eval_y = eval_df.drop(TARGET, axis=1), eval_df[TARGET]
    train_y, eval_y = [np.ravel(x) for x in [train_y, eval_y]]

    return train_x, train_y, eval_x, eval_y
