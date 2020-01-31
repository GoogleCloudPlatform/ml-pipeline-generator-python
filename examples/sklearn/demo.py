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


def main():
    config = "examples/sklearn/config.yaml"
    pred_input = [
        [6.8, 2.8, 4.8, 1.4],
        [6.0, 3.4, 4.5, 1.6],
    ]

    model = SklearnModel(config)
    model.train(cloud=True)
    version = model.serve()
    preds = model.online_predict(pred_input, version=version)
    print("Predictions: {}".format(preds))


if __name__ == "__main__":
    main()
