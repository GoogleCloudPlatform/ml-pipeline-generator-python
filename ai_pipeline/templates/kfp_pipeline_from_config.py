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
"""Kubeflow Pipeline Example."""
import json
import kfp
import kfp.dsl as dsl
from kfp.components import ComponentStore
from kfp.gcp import use_gcp_secret

cs = ComponentStore(local_search_paths=['.', '{{config.output_package}}'],
                    url_search_prefixes=['{{config.github_component_url}}'])
preprocess_op = cs.load_component('{{config.preprocess.component}}')
hpt_op = cs.load_component('hptune')
param_comp = cs.load_component('get_tuned_params')
train_op = cs.load_component('{{config.train.component}}')
deploy_op = cs.load_component('{{config.deploy.component}}')


@dsl.pipeline(
    name='KFP-Pipelines Example',
    description='Kubeflow pipeline generated from ai-pipeline asset'
)
def pipeline_sample(
        project_id='{{config.project_id}}',
        region='{{config.region}}',
        python_module='{{config.train.python_module}}',
        package_uri='{{config.train.python_package}}',
        dataset_bucket='{{config.bucket_id}}',
        staging_bucket='gs://{{config.bucket_id}}',
        job_dir_hptune='gs://{{config.bucket_id}}/hptune',
        job_dir_train='gs://{{config.bucket_id}}/train',
        runtime_version_train='{{config.runtime_version}}',
        runtime_version_deploy='{{config.runtime_version}}',
        hptune_config='{{config.hptune.config}}',
        model_id='{{config.deploy.model_id}}',
        version_id='{{config.deploy.version_id}}',
        common_args_hpt=json.dumps([
           {% for arg in config.hptune.args %}
             {% set name = arg.name %}
             {% set value = arg.default %}
            '--{{name}}', '{{value}}',
           {% endfor %}
            ]),
        common_args_train=json.dumps([
           {% for arg in config.train.args %}
             {% set name =  arg.name %}
             {% set value = arg.default%}
            '--{{name}}', '{{value}}',
           {% endfor %}
            ]),
        replace_existing_version=True):
    """."""
    preprocess_task = preprocess_op(
        {% for arg in config.preprocess.component_args %}
          {% set name = arg.name %}
        {{name}}={{name}},
        {% endfor %}
        )

    hpt_task = hpt_op(
        region=region,
        python_module=python_module,
        package_uri=package_uri,
        staging_bucket=staging_bucket,
        job_dir=job_dir_hptune,
        config=hptune_config,
        runtime_version=runtime_version_train,
        args=common_args_hpt
        )
    hpt_task.after(preprocess_task)

    param_task = param_comp(
        project_id=project_id,
        hptune_job_id=hpt_task.outputs['job_id'].to_struct(),
        common_args=common_args_train
        )

    train_task = train_op(
        project_id=project_id,
        python_module=python_module,
        package_uris=json.dumps([package_uri.to_struct()]),
        region=region,
        args=str(param_task.outputs['tuned_parameters_out']),
        job_dir=job_dir_train,
        python_version='',
        runtime_version=runtime_version_train,
        master_image_uri='',
        worker_image_uri='',
        training_input='',
        job_id_prefix='',
        wait_interval='30'
        )

    deploy_model = deploy_op(  # pylint: disable=unused-variable
        model_uri=train_task.outputs['job_dir'].to_struct()+'{{config.train.model_out_prefix}}',
        project_id=project_id,
        model_id=model_id,
        version_id=version_id,
        runtime_version=runtime_version_deploy,
        replace_existing_version=replace_existing_version
        )

    kfp.dsl.get_pipeline_conf().add_op_transformer(
        use_gcp_secret('user-gcp-sa'))

client = kfp.Client(host='{{config.kfp_deployment_url}}')

client.create_run_from_pipeline_func(pipeline_sample, arguments={})

