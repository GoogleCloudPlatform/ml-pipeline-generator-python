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
from ai_pipeline.models import TFModel
from ai_pipeline.pipelines import KfpPipeline
from examples.preprocess.census_preprocess import load_data


def main():
    config = "examples/tf/config.yaml"
    model = TFModel(config)
    model.generate_files()
    pipeline = KfpPipeline(model)

    # preprocess and upload dataset to expected location.
    load_data(model.data["train"], model.data["evaluation"])

    # define pipeline structure
    p = pipeline.add_train_component()
    pipeline.add_deploy_component(parent=p)
    pipeline.add_predict_component(parent=p)

    pipeline.print_structure()
    pipeline.generate_pipeline()


if __name__ == "__main__":
    main()
