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
"""Model class definitions."""
import os
import stat

import jinja2 as jinja

from ai_pipeline.parsers import parse_yaml


class BaseModel(object):
    """BaseModel class."""

    def __init__(self, config, template):
        self.template = template

        self._set_config(config)
        self._populate_trainer()

    def _set_config(self, config_path):
        """."""
        config = parse_yaml(config_path)
        for key in config:
            setattr(self, key, config[key])

    def _populate_trainer(self):
        """Use Jinja templates to generate model training code."""
        loader = jinja.PackageLoader("ai_pipeline", "templates")
        env = jinja.Environment(loader=loader)

        model_template = env.get_template("sklearn_model.py")
        model_file = model_template.render(model_path=self.model["path"])
        with open("trainer/model.py", "w+") as f:
            f.write(model_file)

        task_template = env.get_template("sklearn_task.py")
        task_file = task_template.render(
            model_name=self.model["name"],
            model_path=self.model["path"],
            args=self.args)
        with open("trainer/task.py", "w+") as f:
            f.write(task_file)

        run_template = env.get_template("run.train.sh")
        run_file = run_template.render(
            project_id=self.project_id,
            bucket_id=self.bucket_id)
        run_file_path = "bin/run.train.sh"
        with open(run_file_path, "w+") as f:
            f.write(run_file)

        st = os.stat(run_file_path)
        os.chmod(run_file_path, st.st_mode | stat.S_IEXEC)

    def train(self):
        pass


class SklearnModel(BaseModel):
    """SklearnModel class."""

    def __init__(self, config):
        super(SklearnModel, self).__init__(config, "sklearn")
