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
    {% filter indent(width=4, indentfirst=False) %}
    params = {{train_params}}
    {% endfilter %}

    # TODO(humichael): Determine if this is needed.
    params["job_id_prefix"] += prev_op_id
    mlengine_train_op = kfp.components.load_component_from_url(
        "{}/ml_engine/train/component.yaml".format(github_url))
    train_op = mlengine_train_op(**params).apply(
        gcp.use_gcp_secret("user-gcp-sa"))
    return train_op


# TODO(humichael): Support best model if HP Tuning.
def get_model_path(prev_op_id="") -> NamedTuple("params", [
        ("model_path", str),
        ("stub", str),
]):
    """Builds a model path prefix to use to search for the export dir."""
    model_path = "{{ model_dir }}"
    return (model_path, prev_op_id)


def get_deploy_op(github_url, prev_op_id=""):
    """Returns an op for running AI Platform batch prediction jobs.

    Args:
      github_url: url to the github commit the component definition will be
        read from.
      prev_op_id: an output from a previous component to use to chain
        components together.

    Returns:
      A Kubeflow Pipelines component for running batch predictions.
    """

    {% filter indent(width=4, indentfirst=False) %}
    params = {{deploy_params}}
    {% endfilter %}

    params["version_id"] = prev_op_id
    if "model_uri" not in params:
        model_path_op = make_op_func(get_model_path)(prev_op_id)
        list_blobs = kfp.components.load_component(
            "orchestration/components/list_blobs.yaml")
        gsutil_op = list_blobs(model_path_op.outputs["model_path"]).apply(
            gcp.use_gcp_secret("user-gcp-sa"))
        params["model_uri"] = gsutil_op.output

    mlengine_batch_predict_op = kfp.components.load_component_from_url(
        "{}/ml_engine/deploy/component.yaml".format(github_url))
    predict_op = mlengine_batch_predict_op(**params).apply(
        gcp.use_gcp_secret("user-gcp-sa"))
    return predict_op


# TODO(humichael): Implement after supporting batch prediction.
def get_predict_op():
    pass


@kfp.dsl.pipeline(
    name="train_pipeline",
    description="Pipeline for training a model on CAIP.")
def train_pipeline():
    """Defines a Kubeflow Pipeline."""
    github_url = ("https://raw.githubusercontent.com/kubeflow/pipelines/"
                  + "02c991dd265054b040265b3dfa1903d5b49df859/components/gcp")

    # TODO(humichael): Add keyword arguments.
    {% for p, c in relations %}
        {% set parent = components[p] %}
        {% set parent_name = "{}_{}_op".format(parent.role, parent.id) %}
        {% set parent_func = "get_{}_op".format(parent.role) %}
        {% set child = components[c] %}
        {% set child_name = "{}_{}_op".format(child.role, child.id) %}
        {% set child_func = "get_{}_op".format(child.role) %}

        {% if p == -1 %}
    {{ child_name }} = {{ child_func }}(github_url)
        {% else %}
    {{ child_name }} = {{ child_func }}(
        github_url,
        prev_op_id={{ parent_name }}.outputs["job_id"],
    )
        {% endif %}
    {% endfor %}


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        train_pipeline, "train_pipeline.tar.gz")
