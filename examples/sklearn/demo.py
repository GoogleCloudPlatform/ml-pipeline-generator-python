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
from examples.preprocess.census_preprocess import load_data


def _upload_data_to_gcs(model):
    load_data(model.data["train"], model.data["evaluation"])


def main():
    config = "examples/sklearn/config.yaml"
    pred_input = [
        [0.02599666, 6, 1.1365801, 4, 0, 1, 4, 0.14693314, -0.21713187,
         -0.034039237, 38],
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
