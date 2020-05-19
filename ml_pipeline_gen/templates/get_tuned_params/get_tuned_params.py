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
"""Script to extract hyperparamters from the job-ID."""
import argparse

from pathlib import Path

from googleapiclient import discovery
from googleapiclient import errors
from types import SimpleNamespace
import ast


# Modified from: https://stackoverflow.com/a/54332748
class NestedNamespace(SimpleNamespace):
    """Parse nested disctionary to create nested namespace object."""

    def __init__(self, dictionary, **kwargs):
        super(NestedNamespace, self).__init__(**kwargs)
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.__setattr__(key, NestedNamespace(value))
            elif isinstance(value, list):
                self.__setattr__(key,
                                 [NestedNamespace(i)
                                  if isinstance(i, dict)
                                  else i for i in value])
            else:
                self.__setattr__(key, value)


def print_best_parameters(project_id,
                          hp_tune_job,
                          filename='tuned_params',
                          common_args='[]'):
    """Select best hyperparameter set from the job_id."""
    job_id = 'projects/{}/jobs/{}'.format(project_id, hp_tune_job)

    # Build a representation of the Cloud ML API.
    ml = discovery.build('ml', 'v1')

    # Create a request to call projects.models.create.
    request = ml.projects().jobs().get(name=job_id)
    # Make the call.
    try:
        response = request.execute()
    except errors.HttpError as err:
        # Something went wrong, print out some information.
        print('There was an error getting the job info, Check the details:')
        print(err._get_reason())

    job_info = NestedNamespace(response)
    param_list = ast.literal_eval(common_args)
    for key, value in job_info.trainingOutput.trials[0].hyperparameters.__dict__.items():
        param_list.append('--'+key)
        param_list.append(value)
    # Creating the directory where the output file will be created (the directory may or may not exist).
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, 'w') as f:
        f.write(str(param_list))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hptune_job_id',
                        type=str,
                        required=True,
                        help='ID of hparam search job')
    parser.add_argument('--project_id',
                        type=str,
                        required=True,
                        help='GCP project ID')
    parser.add_argument('--common_args',
                        type=str,
                        required=True,
                        help='common (not tunable) arguments for training application')
    parser.add_argument('--tuned_parameters_out',
                        type=str,
                        required=True,
                        help='Path to the file containing Tuned Parameters array')
    args = parser.parse_args()
    print_best_parameters(args.project_id, args.hptune_job_id, args.tuned_parameters_out, args.common_args)
