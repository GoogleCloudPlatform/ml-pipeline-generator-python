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
from ml_pipeline_gen.pipelines import KfpPipeline


def main():
    config = './config.yaml'
    pipeline = KfpPipeline(config=config)
    # Review the components
    pipeline.list_components()
    # define pipeline structure
    preprocess = pipeline.add_component('preprocess')
    hptune = pipeline.add_component('hptune', parent=preprocess)
    get_best_params = pipeline.add_component('get_tuned_params', parent=hptune)
    train = pipeline.add_component('train', parent=get_best_params)
    deploy = pipeline.add_component('deploy', parent=train)

    pipeline.print_structure()
    pipeline.generate_pipeline_from_config()


if __name__ == '__main__':
    main()
