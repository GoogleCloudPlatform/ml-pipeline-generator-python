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
"""Method for generating component files from respective templates."""
from os import path
import pathlib

from ml_pipeline_gen.parsers import parse_yaml
import jinja2 as jinja


def generate_component(config, name, template_spec='./component_spec.yaml'):
    """Generate the component files from the templates."""
    template_spec_path = path.join(path.dirname(__file__), template_spec)
    output_spec = parse_yaml(template_spec_path)
    current_spec = output_spec[name]

    loader = jinja.PackageLoader('ml_pipeline_gen', current_spec['template_dir'])
    env = jinja.Environment(loader=loader, trim_blocks=True,
                            lstrip_blocks='True')
    template_file_list = current_spec['files']
    for template in template_file_list:
        template_in = env.get_template(template['input'])
        template_out = template_in.render(

            config=config
        )
        output_file = path.join(config.output_package, template['output'])
        pathlib.Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(template_out)
