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
"""Demo for TF AI Pipeline."""
import json
import os

from ai_pipeline.models import TFModel


def main():
    config = "examples/tf/config.yaml"
    pred_input = [{
        "age": 25,
        "workclass": "Private",
        "education": "11th",
        "education_num": 7,
        "marital_status": "Never-married",
        "occupation": "Machine-op-inspct",
        "relationship": "Own-child",
        "race": " Black",
        "gender": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States",
    }]

    model = TFModel(config)
    job_id = model.train()
    version = model.serve(job_id=job_id)
    preds = model.online_predict(pred_input, version=version)

    print("Online Predictions")
    print("Features: {}".format(pred_input))
    print("Predictions: {}".format(preds))

    input_path = "./tf_input_data.json"
    with open(input_path, "w+") as f:
        for features in pred_input:
            f.write(json.dumps(features) + "\n")
    input_gcs_path = model.upload_pred_input_data(input_path)
    os.remove(input_path)
    model.batch_predict(input_gcs_path, version=version,
                        input_data_format="JSON")

    print("Batch predictions written to", model.get_pred_output_path())


if __name__ == "__main__":
    main()
