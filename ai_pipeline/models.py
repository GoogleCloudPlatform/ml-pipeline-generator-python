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
"""Model class definitions."""
import abc
import datetime as dt
import os
import subprocess
import time

from googleapiclient import discovery
from googleapiclient import errors
import jinja2 as jinja
import tensorflow.compat.v1 as tf

from ai_pipeline.parsers import parse_yaml


class BaseModel(abc.ABC):
    """Abstract class representing an ML model."""

    def __init__(self, config, framework):
        self._set_config(config)
        self.ml_client = discovery.build("ml", "v1")
        self.framework = framework
        # TODO(humichael): Move this to config and generate setup.py
        self.model_dir = "models"
        self.package_name = "ai-pipeline"
        self.use_hpt = self._use_hpt()

    def _set_config(self, config_path):
        """Parses the given config file and sets instance variables accordingly."""
        config = parse_yaml(config_path)
        for key in config:
            setattr(self, key, config[key])
        # TODO(humichael): Validate config

    # TODO(humichael): Move to utils
    def _get_parent(self, model=False, version="", job="",
                    operation=""):
        """Returns the parent to pass to the CAIP API.

        Args:
            model: true if the parent entity is a model.
            version: a version name.
            job: a job id.
            operation: an operation name.

        Returns:
            a parent entity to pass to a CAIP API call. With no additional
            parameters, a project is returned. However, setting any one of the
            keyword args will change the retuned entity based on the set
            parameter.
        """
        parent = "projects/{}".format(self.project_id)
        if version:
            parent += "/models/{}/versions/{}".format(
                self.model["name"], version)
        elif model:
            parent += "/models/{}".format(self.model["name"])
        elif job:
            parent += "/jobs/{}".format(job)
        elif operation:
            parent += "/operations/{}".format(operation)
        return parent

    def _use_hpt(self):
        """Determines if the training step uses hyperparameters."""
        return self.hyperparameter["directory"]

    # TODO(humichael): Move to utils
    def _call_ml_client(self, request, silent_fail=False):
        """Calls the CAIP API by executing the given request.

        Args:
            request: an API request built using self.ml_client.
            silent_fail: does not raise errors if True.

        Returns:
            response: a dict representing either the response of a successful
                call or the error message of an unsuccessful call.
            success: True if the API call was successful.

        Raises:
            HttpError: when API call fails and silent_fail is set to False.
        """
        try:
            response = request.execute()
            success = True
        except errors.HttpError as err:
            if not silent_fail:
                raise err
            # pylint: disable=protected-access
            response = {"error": err._get_reason()}
            success = False
        return response, success

    # TODO(humichael): find way to avoid using relative paths.
    @abc.abstractmethod
    def _populate_trainer(self, task_template_path,
                          model_template_path):
        """Use Jinja templates to generate model training code.

        Args:
            task_template_path: path to task.py template.
            model_template_path: path to model.py template.
        """
        loader = jinja.PackageLoader("ai_pipeline", "templates")
        env = jinja.Environment(loader=loader, trim_blocks=True,
                                lstrip_blocks="True")

        task_template = env.get_template(task_template_path)
        task_file = task_template.render(
            model_name=self.model["name"],
            model_path=self.model["path"],
            args=self.args)
        with open("trainer/task.py", "w+") as f:
            f.write(task_file)

        model_template = env.get_template(model_template_path)
        model_file = model_template.render(
            model_path=self.model["path"])
        with open("trainer/model.py", "w+") as f:
            f.write(model_file)

    def get_model_dir(self):
        """Returns the GCS path to the model dir."""
        return os.path.join(
            "gs://", self.bucket_id, self.model["name"], self.model_dir)

    def get_job_dir(self):
        """Returns the GCS path to the job dir."""
        return os.path.join(
            "gs://", self.bucket_id, self.model["name"])

    def _get_deployment_dir(self, *_):
        """Returns the GCS path to the exported model."""
        return self.get_model_dir()

    def _wait_until_done(self, job_id, wait_interval=60):
        """Blocks until the given job is completed.

        Args:
            job_id: a CAIP job id.
            wait_interval: the amount of seconds to wait after checking the job
                state.

        Raises:
            RuntimeError: if the job does not succeed.
        """
        state = ""
        end_states = ["SUCCEEDED", "FAILED", "CANCELLED"]
        jobs_client = self.ml_client.projects().jobs()

        print(
            "Waiting for {} to complete. Checking every {} seconds.".format(
                job_id, wait_interval))
        while state not in end_states:
            time.sleep(wait_interval)
            request = jobs_client.get(name=self._get_parent(job=job_id))
            response, _ = self._call_ml_client(request)
            state = response["state"]
            print("Job state of {}: {}".format(job_id, state))
        if state != "SUCCEEDED":
            raise RuntimeError(
                "Job didn't succeed. End state: {}".format(state))

    def _get_staging_dir(self):
        """Returns the GCS path to the staging dir."""
        return os.path.join(
            "gs://", self.bucket_id, self.model["name"], "staging")

    # TODO(humichael): Move to utils.py
    def upload_trainer_dist(self):
        """Builds a source distribution and uploads it to GCS."""
        dist_dir = "dist"
        dist_file = "{}-1.0.tar.gz".format(self.package_name)
        staging_dir = self._get_staging_dir()
        subprocess.call(["python", "setup.py", "sdist"],
                        stdout=open(os.devnull, "wb"))
        if not tf.io.gfile.exists(staging_dir):
            tf.io.gfile.makedirs(staging_dir)

        src = os.path.join(dist_dir, dist_file)
        dst = os.path.join(staging_dir, dist_file)
        tf.io.gfile.copy(src, dst, overwrite=True)
        return dst

    def train(self, blocking=True, wait_interval=60):
        """Trains on CAIP.

        Args:
            blocking: true if the function should exit only once the job
                completes.
            wait_interval: if blocking, how often the job state should be
                checked.
        Returns:
            job_id: a CAIP job id.
        """
        now = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        job_id = "{}_{}".format(self.model["name"], now)
        package_uri = self.upload_trainer_dist()
        jobs_client = self.ml_client.projects().jobs()
        body = {
            "jobId": job_id,
            "trainingInput": {
                "scaleTier": self.scale_tier,
                "packageUris": [package_uri],
                "pythonModule": "trainer.task",
                "args": [
                    "--model_dir", self.get_model_dir(),
                ],
                "jobDir": self.get_job_dir(),
                "region": self.region,
                "runtimeVersion": self.runtime_version,
                "pythonVersion": self.python_version,
            },
        }
        # TODO(smhosein): should we handle custom scale tiers?
        if self.use_hpt:
            body["trainingInput"][
                "hyperparameters"] = self._get_hyperparameters()
        request = jobs_client.create(parent=self._get_parent(),
                                     body=body)
        self._call_ml_client(request)
        if blocking:
            self._wait_until_done(job_id, wait_interval)
        return job_id

    def _get_hyperparameters(self):
        """Generates a dictionary of hyperparameter values for the model."""
        hp_config = parse_yaml(self.hyperparameter["directory"])
        hp = hp_config["trainingInput"]["hyperparameters"]
        # TODO(smhosein): replace ifs with hp.get("key", "default value").
        hyperparams = {
            "goal": hp["goal"] if "goal" in hp else "MAXIMIZE",
            "hyperparameterMetricTag": (hp["hyperparameterMetricTag"]
                                        if "hyperparameterMetricTag" in hp
                                        else "accuracy"),
            "maxTrials": hp["maxTrials"] if "maxTrials" in hp else 4,
            "maxParallelTrials": (hp["maxParallelTrials"]
                                  if "maxParallelTrials" in hp else 1),
            "enableTrialEarlyStopping": (hp["enableTrialEarlyStopping"]
                                         if "enableTrialEarlyStopping" in hp
                                         else True),
            "params": [],
        }
        for param in hp["params"]:
            if param["type"] in ("DOUBLE", "INTEGER"):
                hyperparams["params"].append(
                    self._get_double_int_param(param))
            elif param["type"] == "CATEGORICAL":
                hyperparams["params"].append(
                    self._get_cat_distcrete_param(param,
                                                  "categoricalValues"))
            else:
                hyperparams["params"].append(
                    self._get_cat_distcrete_param(param,
                                                  "discreteValues"))

        return hyperparams

    def _get_double_int_param(self, param):
        """Get the values and ranges for an INTEGER or DOUBLE parameter.

        Args:
            param: dictionary containing, a description of the parameter

        Returns:
            A dict of hyperparameters.
        """
        hyperparam = {
            "parameterName": param["parameterName"],
            "type": param["type"],
            "minValue": param["minValue"],
            "maxValue": param["maxValue"],
            "scaleType": param["scaleType"],
        }
        return hyperparam

    def _get_cat_distcrete_param(self, param, name):
        """Get the categories/values for a CATEGORICAL or DISCRETE parameter.

        Args:
            param: dictionary containing, a description of the parameter
            name: string indicating the type of parameter, i.e.
            either categoricalValues or discreteValues

        Returns:
            A dict of hyperparameters.
        """

        hyperparam = {
            "parameterName": param["parameterName"],
            "type": param["type"],
            name: param[name]
        }
        return hyperparam

    def train_local(self):
        """Trains the model locally."""
        subprocess.call("bin/run.local_train.sh")

    def _create_model(self):
        """Creates a model for serving on CAIP."""
        models_client = self.ml_client.projects().models()
        body = {
            "name": self.model["name"],
            "regions": [self.region],
            "onlinePredictionLogging": True,
        }
        request = models_client.create(
            parent=self._get_parent(), body=body)
        self._call_ml_client(request)

    def _wait_until_op_done(self, op_name, wait_interval=30):
        """Blocks until the given Operation is completed.

        Args:
            op_name: a CAIP Operation name.
            wait_interval: the amount of seconds to wait after checking the
                state.
        """
        done = False
        op_client = self.ml_client.projects().operations()

        print(
            "Waiting for {} to complete. Checking every {} seconds.".format(
                op_name, wait_interval))
        while not done:
            time.sleep(wait_interval)
            request = op_client.get(
                name=self._get_parent(operation=op_name))
            response, _ = self._call_ml_client(request)
            done = "done" in response and response["done"]
            print("Operation {} completed: {}".format(op_name, done))

    def _create_version(self, version, job_id, wait_interval=30):
        """Creates a new version of the model for serving.

        Args:
            version: a version number to use to create a version name.
            job_id: a CAIP job id.
            wait_interval: if blocking, how often the job state should be
                checked.

        Returns:
            the name of the version just created.
        """
        versions_client = self.ml_client.projects().models().versions()
        name = "{}_{}".format(self.model["name"], version)
        body = {
            "name": name,
            "deploymentUri": self._get_deployment_dir(job_id),
            "runtimeVersion": self.runtime_version,
            "framework": self.get_deploy_framework(),
            "pythonVersion": self.python_version,
        }
        request = versions_client.create(
            parent=self._get_parent(model=True), body=body)
        op, _ = self._call_ml_client(request)
        op_name = op["name"].split("/")[-1]
        self._wait_until_op_done(op_name, wait_interval)
        return name

    def get_versions(self):
        """Returns the model versions if a model exists.

        Returns:
            response: the API response if a model exists, otherwise an object
                containing the error message.
            model_exists: True if a deployed model exists.
        """
        versions_client = self.ml_client.projects().models().versions()
        request = versions_client.list(
            parent=self._get_parent(model=True))
        response, model_exists = self._call_ml_client(request,
                                                      silent_fail=True)
        return response, model_exists

    @abc.abstractmethod
    def get_deploy_framework(self):
        pass

    # TODO(humichael): Remove this once we've switched to deploy in examples.
    def serve(self, job_id):
        return self.deploy(job_id)

    def deploy(self, job_id):
        """Deploys model and returns the version name created.

        Args:
            job_id: a CAIP job id.

        Returns:
            the name of the version just created.
        """
        response, model_exists = self.get_versions()
        if model_exists:
            if response:
                versions = [int(version["name"].split("_")[-1])
                            for version in response["versions"]]
                version = max(versions) + 1
            else:
                version = 1
        else:
            self._create_model()
            version = 0
        return self._create_version(version, job_id)

    # TODO(humichael): Add option to pass in csv/json file.
    def online_predict(self, inputs, version=""):
        """Uses a deployed model to get predictions for the given inputs.

        Args:
          inputs: a list of feature vectors.
          version: the version name of the deployed model to use. If none is
              provided, the default version will be used.

        Returns:
            a list of predictions.
        """
        name = self._get_parent(model=True)
        if version:
            name = self._get_parent(version=version)
        projects_client = self.ml_client.projects()
        request = projects_client.predict(name=name,
                                          body={"instances": inputs})
        response, _ = self._call_ml_client(request)
        return response["predictions"]

    def batch_predict(self, inputs):
        """Uses a saved model on GCS to make predictions."""
        # [BLOCKED] until train() uses the python API instead of a script.
        # create a Job with PredictionInput
        pass

    # TODO(humichael): clean up with python code, not a shell script.
    def clean_up(self):
        """Delete all generated files."""
        subprocess.call("bin/cleanup.sh")


class SklearnModel(BaseModel):
    """SklearnModel class."""

    def __init__(self, config):
        super(SklearnModel, self).__init__(config, "sklearn")
        self._populate_trainer()

    def _populate_trainer(self):
        super(SklearnModel, self)._populate_trainer(
            "sklearn_task.py", "sklearn_model.py")

    def get_deploy_framework(self):
        return "SCIKIT_LEARN"


class TFModel(BaseModel):
    """TFModel class."""

    def __init__(self, config):
        super(TFModel, self).__init__(config, "tensorflow")
        self._populate_trainer()

    def _populate_trainer(self):
        super(TFModel, self)._populate_trainer("tf_task.py",
                                               "tf_model.py")

    def get_deploy_framework(self):
        return "TENSORFLOW"

    def _get_deployment_dir(self, job_id):
        """Returns the GCS path to the TF exported model.

        Args:
            job_id: a CAIP job id.
        """

        # TODO(smhosein): combine job/model dir so that hpt and normal jobs
        # have a similar path
        if self.use_hpt:
            # TODO(smhosein): if user wants to servre alone make job_id callable
            name = self._get_parent(job=job_id)
            request = self.ml_client.projects().jobs().get(
                name=name).execute()
            best_model = request["trainingOutput"]["trials"][0][
                "trialId"]
            output_path = os.path.join(self.get_job_dir(), best_model,
                                       "export", "exporter")
        else:
            output_path = os.path.join(self.get_model_dir(), "export",
                                       "exporter")
        return str(subprocess.check_output(
            ["gsutil", "ls", output_path]).strip()).split("\\n")[-1].strip("'")


class XGBoostModel(BaseModel):
    """XGBoost class."""

    def __init__(self, config):
        super(XGBoostModel, self).__init__(config, "xgboost")
        self._populate_trainer()

    def _populate_trainer(self):
        super(XGBoostModel, self)._populate_trainer(
            "xgboost_task.py", "xgboost_model.py")

    def get_deploy_framework(self):
        return "XGBOOST"
