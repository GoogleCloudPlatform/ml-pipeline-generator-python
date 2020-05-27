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
"""Demo for TF ML Pipeline Generator."""
import json
import os

from ml_pipeline_gen.models import TFModel
from model.taxi_preprocess import load_data


def _upload_data_to_gcs(model):
    load_data(model.data["train"], model.data["evaluation"])


# TODO(humichael): See if there"s a way to support csv batch predicts.
def _upload_input_data_to_gcs(model, data):
    input_path = "./tf_input_data.json"
    with open(input_path, "w+") as f:
        for features in data:
            f.write(json.dumps(features) + "\n")
    model.upload_pred_input_data(input_path)
    os.remove(input_path)


def main():
    explanations = True
    config = "config.yaml"
    pred_input = [{
        "trip_miles": 1.0,
        "trip_seconds": -0.56447923,
        "fare": -0.5502175,
        "trip_start_month": -1.00234,
        "trip_start_hour": -0.60791147,
        "trip_start_day": 0.38163432,
        "pickup_community_area": 0.5846407,
        "dropoff_community_area": 0.6274534,
        "pickup_census_tract": 1.4543412,
        "dropoff_census_tract": -0.09238409,
        "pickup_latitude": 41.881,
        "pickup_longitude": -87.633,
        "dropoff_latitude": 41.885,
        "dropoff_longitude": -87.62100000000001,
        "payment_type": 1,
        "company": 3
    }]
    model = TFModel(config)
    model.generate_files()
    _upload_data_to_gcs(model)

    job_id = model.train(tune=True)
    version = model.deploy(job_id=job_id, explanations=explanations)
    if explanations:
        explanations = model.online_explanations(pred_input,
                                                 version=version)
        print("Online Explanations")
        print("Explanations: {}".format(explanations))
    preds = model.online_predict(pred_input, version=version)

    print("Online Predictions")
    print("Features: {}".format(pred_input))
    print("Predictions: {}".format(preds))

    if not explanations:
        _upload_input_data_to_gcs(model, pred_input)
        model.batch_predict(version=version)
        print("Batch predictions written to",
              model.get_pred_output_path())


if __name__ == "__main__":
    main()
