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
import abc
import os
import stat

import jinja2 as jinja

from ai_pipeline.parsers import parse_yaml


class BaseModel(abc.ABC):
    """Abstract class representing an ML model."""

    def __init__(self, config, template):
        self.template = template
        self._set_config(config)

    def _set_config(self, config_path):
        """Parses the given config file and sets instance variables accordingly."""
        config = parse_yaml(config_path)
        for key in config:
            setattr(self, key, config[key])

    # TODO(humichael): find way to avoid using relative paths.
    @abc.abstractmethod
    def _populate_trainer(self, task_template_path, model_template_path):
        """Use Jinja templates to generate model training code.

        Args:
            task_template_path: path to task.py template.
            model_template_path: path to model.py template.
        """
        loader = jinja.PackageLoader("ai_pipeline", "templates")
        env = jinja.Environment(loader=loader)

        task_template = env.get_template(task_template_path)
        task_file = task_template.render(
            model_name=self.model["name"],
            model_path=self.model["path"],
            model_type=self.model["type"],
            args=self.args)
        with open("trainer/task.py", "w+") as f:
            f.write(task_file)

        model_template = env.get_template(model_template_path)
        model_file = model_template.render(model_path=self.model["path"])
        with open("trainer/model.py", "w+") as f:
            f.write(model_file)

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
        self._populate_trainer()

    def _populate_trainer(self):
        super(SklearnModel, self)._populate_trainer(
            "sklearn_task.py", "sklearn_model.py")


class TFModel(BaseModel):
    """TFModel class."""

    def __init__(self, config):
        super(TFModel, self).__init__(config, "tf")
        self._populate_trainer()

    def _populate_trainer(self):
        super(TFModel, self)._populate_trainer("tf_task.py", "tf_model.py")

