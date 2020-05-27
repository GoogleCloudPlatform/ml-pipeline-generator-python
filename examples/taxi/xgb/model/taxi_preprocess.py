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


DATA_DIR = os.path.join(tempfile.gettempdir(), "taxi_data")
DATA_URL = ("https://storage.googleapis.com/cloud-samples-data/ml-engine/chicago_taxi/training/small/")
TRAINING_FILE = "taxi_trips_train.csv"
EVAL_FILE = "taxi_trips_eval.csv"
TRAINING_URL = os.path.join(DATA_URL, TRAINING_FILE)
EVAL_URL = os.path.join(DATA_URL, EVAL_FILE)

_CSV_COLUMNS = [
    "tip", "trip_miles", "trip_seconds", "fare", "trip_start_month",
    "trip_start_hour", "trip_start_day", "pickup_community_area", "dropoff_community_area",
    "pickup_census_tract", "dropoff_census_tract", "pickup_latitude", "pickup_longitude",
    "dropoff_latitude", "dropoff_longitude", "payment_type", "company",
]
_LABEL_COLUMN = "tip"

_CATEGORICAL_TYPES = {
    "payment_type": pd.api.types.CategoricalDtype(categories=[
        'No Charge', 'Credit Card', 'Cash', 'Unknown', 'Dispute'
    ]),
    "company": pd.api.types.CategoricalDtype(categories=[
       'Northwest Management LLC', 'Blue Ribbon Taxi Association Inc.',
       'Taxi Affiliation Services', 'Dispatch Taxi Affiliation',
       'Top Cab Affiliation', 'Choice Taxi Association', '5129 - 87128',
       'KOAM Taxi Association', 'Chicago Medallion Leasing INC',
       'Chicago Medallion Management', '3201 - C&D Cab Co Inc',
       '1247 - 72807 Daniel Ayertey', '5776 - Mekonen Cab Company',
       '2092 - 61288 Sbeih company', '0694 - 59280 Chinesco Trans Inc',
       '4197 - Royal Star', 'C & D Cab Co Inc', '3591 - 63480 Chuks Cab',
       '4053 - Adwar H. Nikola', '3141 - Zip Cab',
       '6742 - 83735 Tasha ride inc', '0118 - 42111 Godfrey S.Awir',
       '3385 - Eman Cab', '4053 - 40193 Adwar H. Nikola',
       '3152 - 97284 Crystal Abernathy', '2823 - 73307 Seung Lee',
       '6574 - Babylon Express Inc.', '5724 - 75306 KYVI Cab Inc',
       '5074 - 54002 Ahzmi Inc', '2733 - 74600 Benny Jona',
       '3253 - 91138 Gaither Cab Co.', '3152 - Crystal Abernathy',
       '5437 - Great American Cab Co', '1085 - N and W Cab Co',
       '6488 - 83287 Zuha Taxi', '2192 - 73487 Zeymane Corp',
       '0118 - Godfrey S.Awir', '4197 - 41842 Royal Star',
       '3319 - C&D Cab Company', '4787 - Reny Cab Co',
       '1085 - 72312 N and W Cab Co', "3591- 63480 Chuk's Cab",
       '6743 - 78771 Luhak Corp', '3623-Arrington Enterprises',
       '3623 - 72222 Arrington Enterprises', '3141 - 87803 Zip Cab',
       '5074 - Ahzmi Inc', '3897 - Ilie Malec', '2092 - Sbeih company',
       '6057 - 24657 Richard Addo', '5006 - 39261 Salifu Bawa',
       '3620 - David K. Cab Corp.', '3556 - 36214 RC Andrews Cab',
       '2733 - Benny Jona', '4615 - 83503 Tyrone Henderson',
       '5129 - 98755 Mengisti Taxi', '5724 - 72965 KYVI Cab Inc',
       '585 - 88805 Valley Cab Co', '5997 - 65283 AW Services Inc.',
       '2809 - 95474 C & D Cab Co Inc.', '6743 - Luhak Corp',
       '5874 - 73628 Sergey Cab Corp.', '3897 - 57856 Ilie Malec',
       '3319 - CD Cab Co', '6747 - Mueen Abdalla']),
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

    train_df = pd.read_csv(training_file_path)
    eval_df = pd.read_csv(eval_file_path)

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

