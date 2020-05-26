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
"""Demo for XGBoost ML Pipeline Generator."""
from ml_pipeline_gen.models import XGBoostModel
from model.census_preprocess import load_data


def _upload_data_to_gcs(model):
    load_data(model.data["train"], model.data["evaluation"])


def main():
    config = "config.yaml"
    pred_input = [[
         7.65000000e+02, 2.81400000e+04, 0.00000000e+00, 1.00000000e+00,
         8.30000000e+01, 3.26000000e+05, 8.30000000e+01, 4.87500000e+00,
         3.60000000e+02, 1.00000000e+00, 3.09730330e+05, 3.25000000e+05,
         1.52696700e+04, 4.67629611e+03, 0.00000000e+00, 3.17866362e+05,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 4.87500000e+00, 4.87500000e+00,
         0.00000000e+00, 4.87500000e+00, 0.00000000e+00, 4.87500000e+00,
         0.00000000e+00, 5.95836265e-06, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 2.63157895e-02, 9.99000000e+02, 9.99000000e+02,
         9.99000000e+02, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00,
         1.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 1.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 1.00000000e+00, 1.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00
    ]]

    model = XGBoostModel(config)
    model.generate_files()
    _upload_data_to_gcs(model)

    job_id = model.train()
    version = model.deploy(job_id=job_id)
    preds = model.online_predict(pred_input, version=version)

    print("Features: {}".format(pred_input))
    print("Predictions: {}".format(preds))

if __name__ == "__main__":
    main()
