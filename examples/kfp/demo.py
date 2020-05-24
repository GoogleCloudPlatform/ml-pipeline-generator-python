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
from ml_pipeline_gen.models import TFModel
from ml_pipeline_gen.pipelines import KfpPipeline
from examples.preprocess.census_preprocess import load_data


# pylint: disable=g-import-not-at-top
def main():
    config = "examples/tf/config.yaml"
    model = TFModel(config)
    model.generate_files()
    pipeline = KfpPipeline(model)
    pipeline.configure_kfp_cluster()

    # Preprocess and upload dataset to expected location.
    load_data(model.data["train"], model.data["evaluation"])

    # Define pipeline structure.
    p = pipeline.add_train_component()
    pipeline.add_deploy_component(parent=p)
    pipeline.add_predict_component(parent=p)

    pipeline.generate_pipeline()

    # pylint: disable=import-outside-toplevel
    from orchestration import pipeline as kfp_pipeline
    kfp_pipeline.main()


if __name__ == "__main__":
    main()
