# python3
# Copyright 2020 Google LLC
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
"""Defines a KubeFlow pipeline."""

import kfp
import kfp.gcp as gcp
from typing import NamedTuple


# pylint: disable=redefined-outer-name
# pylint: disable=g-import-not-at-top
# pylint: disable=reimported
def make_op_func(func):
    """Converts a self-contained python function into an op.

    Args:
      func: a python function with no outside dependencies.

    Returns:
      A function that ingests PipelineParams, parses them, and passes the
      results to the given function, all within a container.
    """
    return kfp.components.func_to_container_op(func)


def get_train_op(github_url, prev_op_id=""):
    """Returns an op for running AI Platform training jobs.

    Args:
      github_url: url to the github commit the component definition will be
        read from.
      prev_op_id: an output from a previous component to use to chain
        components together.

    Returns:
      A Kubeflow Pipelines component for running training.
    """
    params = {
        "project_id": "cchatterjee-sandbox",
        "job_id_prefix": "train_tf_model_demo_v1_200607_Jun061591580638",
        "training_input": {
            "scaleTier": "STANDARD_1",
            "packageUris": [
                "gs://cchatterjee-mlpg/tf_model_demo_v1/staging/cchatterjee-mlpg-1.0.tar.gz"
            ],
            "pythonModule": "trainer.task",
            "args": [
                "--model_dir",
                "gs://cchatterjee-mlpg/tf_model_demo_v1/models"
            ],
            "jobDir": "gs://cchatterjee-mlpg/tf_model_demo_v1",
            "region": "us-central1",
            "runtimeVersion": "1.15",
            "pythonVersion": "3.7"
        }
    }

    params["job_id_prefix"] += prev_op_id
    mlengine_train_op = kfp.components.load_component_from_url(
        "{}/ml_engine/train/component.yaml".format(github_url))
    train_op = mlengine_train_op(**params)
    return train_op


def get_model_path(prev_op_id="") -> NamedTuple("params", [
        ("model_path", str),
        ("stub", str),
]):
    """Builds a model path prefix to use to search for the export dir."""
    model_path = "gs://cchatterjee-mlpg/tf_model_demo_v1/models/1/export/export"
    return (model_path, prev_op_id)


def get_model_path_op(prev_op_id):
    """Returns a component for getting the model path."""
    model_path_op = make_op_func(get_model_path)(prev_op_id)
    list_blobs = kfp.components.load_component(
        "orchestration/components/list_blobs.yaml")
    gsutil_op = list_blobs(model_path_op.outputs["model_path"])
    return gsutil_op


def get_deploy_op(github_url, prev_op_id=""):
    """Returns an op for deploying models on CAIP.

    Args:
      github_url: url to the github commit the component definition will be
        read from.
      prev_op_id: an output from a previous component to use to chain
        components together.

    Returns:
      A Kubeflow Pipelines component for deploying models.
    """

    params = {
        "project_id": "cchatterjee-sandbox",
        "model_id": "tf_model_demo_v1_kfp",
        "runtime_version": "1.15",
        "python_version": "3.7"
    }

    params["version_id"] = prev_op_id
    if "model_uri" not in params:
        gsutil_op = get_model_path_op(prev_op_id)
        params["model_uri"] = gsutil_op.output

    mlengine_deploy_op = kfp.components.load_component_from_url(
        "{}/ml_engine/deploy/component.yaml".format(github_url))
    deploy_op = mlengine_deploy_op(**params)
    return deploy_op


def get_predict_op(github_url, prev_op_id="", version_name=""):
    """Returns an op for running AI Platform batch prediction jobs.

    Args:
      github_url: url to the github commit the component definition will be
        read from.
      prev_op_id: an output from a previous component to use to chain
        components together.
      version_name: a version name of a deployed model to predict with.

    Returns:
      A Kubeflow Pipelines component for running batch predictions.
    """

    params = {
        "project_id": "cchatterjee-sandbox",
        "model_path": "projects/cchatterjee-sandbox/models/tf_model_demo_v1",
        "input_paths": [
            "gs://cchatterjee-mlpg/tf_model_demo_v1/inputs/*"
        ],
        "input_data_format": "JSON",
        "output_path": "gs://cchatterjee-mlpg/tf_model_demo_v1/outputs",
        "region": "us-central1",
        "output_data_format": "JSON",
        "job_id_prefix": "train_tf_model_demo_v1_200607_Jun061591580638"
    }

    if prev_op_id:
        gsutil_op = get_model_path_op(prev_op_id)
        params["model_path"] = gsutil_op.output
    elif version_name:
        params["model_path"] = version_name
    mlengine_batch_predict_op = kfp.components.load_component_from_url(
        "{}/ml_engine/batch_predict/component.yaml".format(github_url))
    predict_op = mlengine_batch_predict_op(**params)
    return predict_op


@kfp.dsl.pipeline(
    name="train_pipeline",
    description="Pipeline for training a model on CAIP.")
def train_pipeline():
    """Defines a Kubeflow Pipeline."""
    github_url = ("https://raw.githubusercontent.com/kubeflow/pipelines/"
                  + "02c991dd265054b040265b3dfa1903d5b49df859/components/gcp")

    # TODO(humichael): Add params.

    train_0_op = get_train_op(github_url)

    deploy_1_op = get_deploy_op(
        github_url,
        prev_op_id=train_0_op.outputs["job_id"],
    )

    predict_2_op = get_predict_op(
        github_url,
        prev_op_id=train_0_op.outputs["job_id"],
    )


def main(compile=False):
    """Compile the pipeline and also create a run."""
    if compile:
        kfp.compiler.Compiler().compile(train_pipeline, "train_pipeline.tar.gz")

    client = kfp.Client(host="1ae3e621a1650575-dot-us-central2.pipelines.googleusercontent.com")
    client.create_run_from_pipeline_func(train_pipeline, arguments={})


if __name__ == "__main__":
    main()