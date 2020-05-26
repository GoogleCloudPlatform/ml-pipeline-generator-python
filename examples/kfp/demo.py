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
"""Demo for KubeFlow Pipelines."""
import json
import os

from ml_pipeline_gen.models import TFModel
from ml_pipeline_gen.pipelines import KfpPipeline
from examples.preprocess.census_preprocess import load_data


def _upload_data_to_gcs(model):
    """Calls the preprocessing fn which uploads train/eval data to GCS."""
    load_data(model.data["train"], model.data["evaluation"])


# TODO(humichael): See if there's a way to support csv batch predicts.
def _upload_input_data_to_gcs(model, data):
    input_path = "tf_input_data.json"
    with open(input_path, "w+") as f:
        for features in data:
            f.write(json.dumps(features) + "\n")
    model.upload_pred_input_data(input_path)
    os.remove(input_path)


# pylint: disable=g-import-not-at-top
def main():
    config = "examples/tf/config.yaml"
    model = TFModel(config)
    model.generate_files()
    _upload_data_to_gcs(model)
    pipeline = KfpPipeline(model)

    # preprocess and upload dataset to expected location.
    load_data(model.data["train"], model.data["evaluation"])

    # define pipeline structure
    p = pipeline.add_train_component()
    pipeline.add_deploy_component(parent=p)
    pipeline.add_predict_component(parent=p)
    pipeline.print_structure()

    pipeline.generate_pipeline()

    # Create batch prediction data in GCS.
    pred_input = [{
        "age": 0.02599666,
        "workclass": 6,
        "education_num": 1.1365801,
        "marital_status": 4,
        "occupation": 0,
        "relationship": 1,
        "race": 4,
        "capital_gain": 0.14693314,
        "capital_loss": -0.21713187,
        "hours_per_week": -0.034039237,
        "native_country": 38,
        "income_bracket": 0,
    }]
    _upload_input_data_to_gcs(model, pred_input)

    # Run the pipeline.
    # pylint: disable=import-outside-toplevel
    from orchestration import pipeline as kfp_pipeline
    kfp_pipeline.main()


if __name__ == "__main__":
    main()
