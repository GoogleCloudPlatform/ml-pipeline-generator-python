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
"""Train a simple TF classifier for MNIST dataset.

This example comes from the cloudml-samples keras demo.
github.com/GoogleCloudPlatform/cloudml-samples/blob/master/census/tf-keras
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import urllib
import tempfile

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf


DATA_DIR = os.path.join(tempfile.gettempdir(), "census_data")
DATA_URL = ("https://storage.googleapis.com/cloud-samples-data/ai-platform"
            + "/census/data/")
TRAINING_FILE = "adult.data.csv"
EVAL_FILE = "adult.test.csv"
TRAINING_URL = os.path.join(DATA_URL, TRAINING_FILE)
EVAL_URL = os.path.join(DATA_URL, EVAL_FILE)

_CSV_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "gender",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income_bracket",
]
_LABEL_COLUMN = "income_bracket"
UNUSED_COLUMNS = ["fnlwgt", "education", "gender"]

_CATEGORICAL_TYPES = {
    "workclass": pd.api.types.CategoricalDtype(categories=[
        "Federal-gov", "Local-gov", "Never-worked", "Private", "Self-emp-inc",
        "Self-emp-not-inc", "State-gov", "Without-pay"
    ]),
    "marital_status": pd.api.types.CategoricalDtype(categories=[
        "Divorced", "Married-AF-spouse", "Married-civ-spouse",
        "Married-spouse-absent", "Never-married", "Separated", "Widowed"
    ]),
    "occupation": pd.api.types.CategoricalDtype([
        "Adm-clerical", "Armed-Forces", "Craft-repair", "Exec-managerial",
        "Farming-fishing", "Handlers-cleaners", "Machine-op-inspct",
        "Other-service", "Priv-house-serv", "Prof-specialty", "Protective-serv",
        "Sales", "Tech-support", "Transport-moving"
    ]),
    "relationship": pd.api.types.CategoricalDtype(categories=[
        "Husband", "Not-in-family", "Other-relative", "Own-child", "Unmarried",
        "Wife"
    ]),
    "race": pd.api.types.CategoricalDtype(categories=[
        "Amer-Indian-Eskimo", "Asian-Pac-Islander", "Black", "Other", "White"
    ]),
    "native_country": pd.api.types.CategoricalDtype(categories=[
        "Cambodia", "Canada", "China", "Columbia", "Cuba", "Dominican-Republic",
        "Ecuador", "El-Salvador", "England", "France", "Germany", "Greece",
        "Guatemala", "Haiti", "Holand-Netherlands", "Honduras", "Hong",
        "Hungary", "India", "Iran", "Ireland", "Italy", "Jamaica", "Japan",
        "Laos", "Mexico", "Nicaragua", "Outlying-US(Guam-USVI-etc)", "Peru",
        "Philippines", "Poland", "Portugal", "Puerto-Rico", "Scotland", "South",
        "Taiwan", "Thailand", "Trinadad&Tobago", "United-States", "Vietnam",
        "Yugoslavia"
    ]),
    "income_bracket": pd.api.types.CategoricalDtype(categories=[
        "<=50K", ">50K"
    ])
}


def _download_and_clean_file(filename, url):
    """Downloads data from url, and makes changes to match the CSV format.

    The CSVs may use spaces after the comma delimters (non-standard) or include
    rows which do not represent well-formed examples. This function strips out
    some of these problems.

    Args:
      filename: filename to save url to
      url: URL of resource to download
    """
    temp_file, _ = urllib.request.urlretrieve(url)
    with tf.io.gfile.GFile(temp_file, "r") as temp_file_object:
        with tf.io.gfile.GFile(filename, "w") as file_object:
            for line in temp_file_object:
                line = line.strip()
                line = line.replace(", ", ",")
                if not line or "," not in line:
                    continue
                if line[-1] == ".":
                    line = line[:-1]
                line += "\n"
                file_object.write(line)
    tf.io.gfile.remove(temp_file)


def download(data_dir):
    """Downloads census data if it is not already present.

    Args:
      data_dir: directory where we will access/save the census data

    Returns:
      foo
    """
    tf.io.gfile.makedirs(data_dir)

    training_file_path = os.path.join(data_dir, TRAINING_FILE)
    if not tf.io.gfile.exists(training_file_path):
        _download_and_clean_file(training_file_path, TRAINING_URL)

    eval_file_path = os.path.join(data_dir, EVAL_FILE)
    if not tf.io.gfile.exists(eval_file_path):
        _download_and_clean_file(eval_file_path, EVAL_URL)

    return training_file_path, eval_file_path


def upload(train_df, eval_df, train_path, eval_path):
    train_df.to_csv(os.path.join(os.path.dirname(train_path), TRAINING_FILE),
                    index=False, header=False)
    eval_df.to_csv(os.path.join(os.path.dirname(eval_path), EVAL_FILE),
                   index=False, header=False)


def preprocess(dataframe):
    """Converts categorical features to numeric. Removes unused columns.

    Args:
      dataframe: Pandas dataframe with raw data

    Returns:
      Dataframe with preprocessed data
    """
    dataframe = dataframe.drop(columns=UNUSED_COLUMNS)

    # Convert integer valued (numeric) columns to floating point
    numeric_columns = dataframe.select_dtypes(["int64"]).columns
    dataframe[numeric_columns] = dataframe[numeric_columns].astype("float32")

    # Convert categorical columns to numeric
    cat_columns = dataframe.select_dtypes(["object"]).columns
    dataframe[cat_columns] = dataframe[cat_columns].apply(
        lambda x: x.astype(_CATEGORICAL_TYPES[x.name]))
    dataframe[cat_columns] = dataframe[cat_columns].apply(
        lambda x: x.cat.codes)
    return dataframe


def standardize(dataframe):
    """Scales numerical columns using their means and standard deviation.

    Args:
      dataframe: Pandas dataframe

    Returns:
      Input dataframe with the numerical columns scaled to z-scores
    """
    dtypes = list(zip(dataframe.dtypes.index, map(str, dataframe.dtypes)))
    for column, dtype in dtypes:
        if dtype == "float32":
            dataframe[column] -= dataframe[column].mean()
            dataframe[column] /= dataframe[column].std()
    return dataframe


def load_data(train_path="", eval_path=""):
    """Loads data into preprocessed (train_x, train_y, eval_y, eval_y) dataframes.

    Args:
      train_path: Local or GCS path to uploaded train data to.
      eval_path: Local or GCS path to uploaded eval data to.

    Returns:
      A tuple (train_x, train_y, eval_x, eval_y), where train_x and eval_x are
      Pandas dataframes with features for training and train_y and eval_y are
      numpy arrays with the corresponding labels.
    """
    # Download Census dataset: Training and eval csv files.
    training_file_path, eval_file_path = download(DATA_DIR)

    train_df = pd.read_csv(
        training_file_path, names=_CSV_COLUMNS, na_values="?")
    eval_df = pd.read_csv(eval_file_path, names=_CSV_COLUMNS, na_values="?")

    train_df = preprocess(train_df)
    eval_df = preprocess(eval_df)

    # Split train and eval data with labels. The pop method copies and removes
    # the label column from the dataframe.
    train_x, train_y = train_df, train_df.pop(_LABEL_COLUMN)
    eval_x, eval_y = eval_df, eval_df.pop(_LABEL_COLUMN)

    # Join train_x and eval_x to normalize on overall means and standard
    # deviations. Then separate them again.
    all_x = pd.concat([train_x, eval_x], keys=["train", "eval"])
    all_x = standardize(all_x)
    train_x, eval_x = all_x.xs("train"), all_x.xs("eval")

    # Rejoin features and labels and upload to GCS.
    if train_path and eval_path:
        train_df = train_x.copy()
        train_df[_LABEL_COLUMN] = train_y
        eval_df = eval_x.copy()
        eval_df[_LABEL_COLUMN] = eval_y
        upload(train_df, eval_df, train_path, eval_path)

    # Reshape label columns for use with tf.data.Dataset
    train_y = np.asarray(train_y).astype("float32").reshape((-1, 1))
    eval_y = np.asarray(eval_y).astype("float32").reshape((-1, 1))

    return train_x, train_y, eval_x, eval_y

