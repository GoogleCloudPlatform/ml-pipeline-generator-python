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
"""Utility functions."""
import os

import tensorflow as tf

from sklearn.externals import joblib
from google.cloud import storage



def dump_object(obj, output_path, model_type=""):
    """Pickle the given object and write to output_path.

    Args:
      obj: object to pickle.
      output_path: a local or GCS path.
      model_type: whether we are saving a TF model or sklearn/xgboost
    """
    if not tf.io.gfile.exists(output_path):
        tf.io.gfile.makedirs(os.path.dirname(output_path))
    if model_type == "tf":
        tf.saved_model.save(obj, output_path)
    else:
        with tf.io.gfile.GFile(output_path, "w+") as f:
            joblib.dump(obj, f)


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )