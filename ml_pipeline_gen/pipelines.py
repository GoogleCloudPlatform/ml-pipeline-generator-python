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
"""Pipeline class definitions."""
import abc
import json
import os
from os import path
import subprocess
import pathlib
import time
import datetime as dt
import jinja2 as jinja

from google.cloud import container_v1
from kubernetes import client, config
from kubernetes.client.rest import ApiException
from ml_pipeline_gen.parsers import NestedNamespace
from ml_pipeline_gen.parsers import parse_yaml


class _Component(object):
    """A BasePipeline component (behaves like a tree)."""

    def __init__(self, role, comp_id, params=None):
        self.role = role
        self.id = comp_id
        # TODO(humichael): support children reading parent's params.
        self.params = params if params else {}
        self.children = []

    def add_child(self, comp):
        self.children.append(comp)


class BasePipeline(abc.ABC):
    """Abstract class representing an ML pipeline."""

    def __init__(self, model=None, config=None):
        self.model = model
        self.structure = _Component("start", -1)
        self.size = 0
        if config:
            self.config = NestedNamespace(parse_yaml(config))
        else:
            self.config = config
        if self.model:
            now = dt.datetime.now().strftime("%y%m%d_%h%m%s")
            self.job_id = "{}_{}".format(self.model.model["name"], now)

    def list_components(self):
        all_components = []
        if self.config is not None:
            for k, v in self.config.__dict__.items():
                if hasattr(v, "component"):
                    all_components.append(k)
        print(all_components)

    def add_train_component(self, parent=None, wait_interval=None):
        """Adds a train component after the specified parent."""
        if not parent:
            parent = self.structure
        params = {
            "wait_interval": wait_interval,
        }
        params = {k: v for k, v in params.items() if v is not None}

        component = _Component("train", self.size, params=params)
        parent.add_child(component)
        self.size += 1
        return component

    def add_deploy_component(self, parent=None, model_uri=None,
                             wait_interval=None):
        """Adds a deploy component after the specified parent."""
        if not parent:
            parent = self.structure
        params = {
            "model_uri": model_uri,
            "wait_interval": wait_interval,
        }
        params = {k: v for k, v in params.items() if v is not None}

        component = _Component("deploy", self.size, params=params)
        parent.add_child(component)
        self.size += 1

        return component

    def add_predict_component(self, parent=None, version=None,
                              wait_interval=None):
        """Adds a predict component after the specified parent."""
        if not parent:
            parent = self.structure
        params = {
            "version": version,
            "wait_interval": wait_interval,
        }
        params = {k: v for k, v in params.items() if v is not None}

        component = _Component("predict", self.size, params=params)
        parent.add_child(component)
        self.size += 1
        return component

    def print_structure(self):
        """Prints the structure of the pipeline."""
        next_comps = [self.structure]
        while next_comps:
            comp = next_comps.pop()
            if comp.id != -1:
                print(comp.id, [x.id for x in comp.children])
            next_comps.extend(comp.children)

    def to_graph(self):
        """Represents the pipeline as edges and vertices.

        Returns:
            components: the vertices of the graph.
            relations: the edges of the graph in (parent, child) form.
        """
        components = [None] * self.size
        relations = []
        next_comps = [self.structure]
        while next_comps:
            comp = next_comps.pop()
            next_comps.extend(comp.children)
            if comp.id != -1:
                components[comp.id] = comp
            for child in comp.children:
                relations.append((comp.id, child.id))
        return components, relations

    @abc.abstractmethod
    def generate_pipeline(self):
        """Creates the files to compile a pipeline."""
        pass


class KfpPipeline(BasePipeline):
    """KubeFlow Pipelines class."""

    def __init__(self, model=None, config=None):
        super().__init__(model, config)
        if not self.check_cluster_label("mlpg_wi_auth"):
            self.setup_auth()
            self.update_hostname()

    def setup_auth(self):
        """Calls shell script to verify required auth for KFP cluster."""
        model = self.model
        subprocess.call([
            "bin/wi_setup.sh",
            model.project_id,
            model.cluster_name,
            model.cluster_zone,
            # TODO(ashokpatelapk): Check if namespace can be a config var.
            "default"
        ])

    def update_hostname(self):
        """Returns Hostname (URL) of KFP cluster in current kube context."""
        # Checks default kubectl context from ~/.kube/config
        config.load_kube_config()
        name = "inverse-proxy-config"
        namespace = "default"
        instance = client.CoreV1Api()
        response = instance.read_namespaced_config_map(name, namespace)
        while response.data is None:
            print("Waiting for KFP Dashboard to be updated...")
            time.sleep(10)
            try:
                response = instance.read_namespaced_config_map(name, namespace)
            except ApiException as e:
                print("Exception -> CoreV1Api: {}}".format(e))
        print("Waiting for KFP Dashboard to be updated...")
        time.sleep(30)
        self.model.orchestration["host"] = response.data["Hostname"]

    def check_cluster_label(self, label):
        """Checks a specifed resourceLabel for a GKE cluster"""
        model = self.model
        client = container_v1.ClusterManagerClient()
        cluster_name = "projects/{0}/locations/{1}/clusters/{2}".format(
            model.project_id,
            model.cluster_zone,
            model.cluster_name
        )
        response = client.get_cluster(name=cluster_name)
        return response.resource_labels[label] == "true"

    def _get_train_params(self):
        """Returns parameters for training on CAIP."""
        model = self.model
        package_uri = model.upload_trainer_dist()
        params = {
            "project_id": model.project_id,
            "job_id_prefix": "train_{}".format(self.job_id),
            "training_input": {
                "scaleTier": model.scale_tier,
                "packageUris": [package_uri],
                "pythonModule": "trainer.task",
                "args": [
                    "--model_dir", model.get_model_dir(),
                ],
                "jobDir": model.get_job_dir(),
                "region": model.region,
                "runtimeVersion": model.runtime_version,
                "pythonVersion": model.python_version,
            },
        }
        return json.dumps(params, indent=4)

    def _get_deploy_params(self):
        """Returns parameters for deploying on CAIP."""
        model = self.model
        params = {
            "project_id": model.project_id,
            "model_id": "{}_kfp".format(model.model["name"]),
            "runtime_version": model.runtime_version,
            "python_version": model.python_version,
        }

        if model.framework != "tensorflow":
            params["model_uri"] = model.get_model_dir()
        return json.dumps(params, indent=4)

    def _get_predict_params(self):
        """Returns parameters for predicting on CAIP."""
        model = self.model
        if not model.supports_batch_predict():
            raise RuntimeError("Batch predict not supported for this model.")
        pred_info = model.data["prediction"]
        inputs = pred_info["input_data_paths"]
        if not isinstance(inputs, list):
            inputs = [inputs]
        input_format = (pred_info["input_format"] if "input_format" in pred_info
                        else "DATA_FORMAT_UNSPECIFIED")
        output_format = (pred_info["output_format"]
                         if "output_format" in pred_info else "JSON")
        params = {
            "project_id": model.project_id,
            "model_path": model.get_parent(model=True),
            "input_paths": inputs,
            "input_data_format": input_format,
            "output_path": model.get_pred_output_path(),
            "region": model.region,
            "output_data_format": output_format,
            "job_id_prefix": "train_{}".format(self.job_id),
        }
        return json.dumps(params, indent=4)

    def _write_template(self, env, template_path, args, dest):
        template = env.get_template(template_path)
        body = template.render(**args)
        with open(dest, "w+") as f:
            f.write(body)

    def generate_pipeline(self):
        """Creates the files to compile a pipeline."""
        loader = jinja.PackageLoader("ml_pipeline_gen", "templates")
        env = jinja.Environment(loader=loader, trim_blocks=True,
                                lstrip_blocks="True")
        components, relations = self.to_graph()

        model = self.model
        model_dir = model.get_model_dir()
        if model.framework == "tensorflow":
            # TODO(humichael): Need custom component to get best model.
            # VS: The componet is available
            model_dir = os.path.join(model_dir, "1", "export", "export")

        pipeline_args = {
            "train_params": self._get_train_params(),
            "model_dir": model_dir,
            "deploy_params": self._get_deploy_params(),
            "prediction_params": self._get_predict_params(),
            "components": components,
            "relations": relations,
            "host": model.orchestration["host"],
        }
        self._write_template(env, "kfp_pipeline.py", pipeline_args,
                             "orchestration/pipeline.py")

    def generate_pipeline_from_config(self):
        """Creates the files to compile a pipeline from config file."""
        template_files = [
            ("kfp_pipeline_from_config.py", "orchestration/pipeline.py"),
            ("example_pipeline.ipynb", "orchestration/pipeline.ipynb")
        ]
        loader = jinja.PackageLoader("ml_pipeline_gen", "templates")
        env = jinja.Environment(loader=loader, trim_blocks=True,
                                lstrip_blocks="True")
        for in_file, out_file in template_files:
            pipeline_template = env.get_template(in_file)
            pipeline_file = pipeline_template.render(
                config=self.config,
            )
            output_file = path.join(self.config.output_package, out_file)
            pathlib.Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w+") as f:
                f.write(pipeline_file)
