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
"""Demo for scikit-learn AI Pipeline."""
from ai_pipeline.models import SklearnModel
from examples.preprocess.taxi_preprocess import load_data


def _upload_data_to_gcs(model):
    load_data(model.data["train"], model.data["evaluation"])


def main():
    config = "examples/taxi/sklearn/config.yaml"
    pred_input = [
        [1.0, -0.56447923, -0.5502175, -1.00234, -0.60791147,
         0.38163432,0.5846407, 0.6274534, 1.4543412, -0.09238409,
         41.881, -87.633, 41.885, -87.62100000000001, 1, 3, ],
    ]
    model = SklearnModel(config)
    model.generate_files()
    _upload_data_to_gcs(model)

    job_id = model.train(tune=True)
    version = model.deploy(job_id=job_id)
    preds = model.online_predict(pred_input, version=version)

    print("Features: {}".format(pred_input))
    print("Predictions: {}".format(preds))


if __name__ == "__main__":
    main()
