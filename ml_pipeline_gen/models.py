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
import collections.abc
import datetime as dt
import json
import os
import pathlib
import shutil
import subprocess
import time

from googleapiclient import discovery
from googleapiclient import errors
import jinja2 as jinja
import tensorflow.compat.v1 as tf
from tensorflow.python.tools import saved_model_utils

from ml_pipeline_gen.parsers import parse_yaml


class BaseModel(abc.ABC):
    """Abstract class representing an ML model."""

    def __init__(self, config_path, framework):
        config = parse_yaml(config_path)
        self._set_config(config)
        self.ml_client = discovery.build("ml", "v1")
        self.framework = framework

    def _get_default_config(self):
        return {
            "model": {
                "metrics": [],
            },
        }

    def _deep_update(self, d, u):
        """Updates a dict and any nested dicts within it."""
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = self._deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def _set_config(self, new_config):
        """Iterates through the config dict and sets instance variables."""
        config = self._get_default_config()
        config = self._deep_update(config, new_config)
        for key in config:
            setattr(self, key, config[key])
        self._set_model_params(config)
        # TODO(humichael): Validate config (required, metrics is one of...)

    def _get_default_input_args(self, train_path, eval_path):
        return {
            "train_path": {
                "type": "str",
                "help": "Dir or bucket containing training data.",
                "default": train_path,
            },
            "eval_path": {
                "type": "str",
                "help": "Dir or bucket containing eval data.",
                "default": eval_path,
            },
            "model_dir": {
                "type": "str",
                "help": "Dir or bucket to save model files.",
                "default": "models",
            },
        }

    def _set_model_params(self, config):
        """Sets the input args and updates self.model_dir."""
        model_params = (config["model_params"]
                        if "model_params" in config else {})
        input_args = self._get_default_input_args(
            train_path=config["data"]["train"],
            eval_path=config["data"]["evaluation"],
        )

        if "input_args" in model_params:
            new_input_args = model_params["input_args"]
            input_args = self._deep_update(input_args, new_input_args)
        self.model_params = model_params
        self.model_params["input_args"] = input_args
        self.model_dir = input_args["model_dir"]["default"]

    # TODO(humichael): Move to utils
    def get_parent(self, model=False, version="", job="",
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

    def _write_template(self, env, template_path, args, dest):
        template = env.get_template(template_path)
        body = template.render(**args)
        with open(dest, "w+") as f:
            f.write(body)

    def _write_static(self):
        """Copies static files to the working directory."""
        root_dir = pathlib.Path(__file__).parent.resolve()
        for d in root_dir.joinpath("static").iterdir():
            if d.is_dir() and not os.path.exists(d.stem):
                shutil.copytree(d, d.stem)

    # TODO(humichael): find way to avoid using relative paths.
    @abc.abstractmethod
    def generate_files(self, task_template_path,
                       model_template_path, inputs_template_path):
        """Use Jinja templates to generate model training code.

        Args:
            task_template_path: path to task.py template.
            model_template_path: path to model.py template.
            inputs_template_path: path to inputs.py template.
        """
        loader = jinja.PackageLoader("ml_pipeline_gen", "templates")
        env = jinja.Environment(loader=loader, trim_blocks=True,
                                lstrip_blocks="True")

        task_args = {
            "input_args": self.model_params["input_args"],
        }
        model_args = {
            "model_path": self.model["path"],
            "metrics": self.model["metrics"],
        }
        inputs_args = {
            "schema": json.dumps(self.data["schema"], indent=4),
            "target": self.model["target"],
        }
        setup_args = {"package_name": self.package_name}

        self._write_static()
        self._write_template(env, task_template_path, task_args,
                             "trainer/task.py")
        self._write_template(env, model_template_path, model_args,
                             "trainer/model.py")
        self._write_template(env, inputs_template_path, inputs_args,
                             "trainer/inputs.py")
        self._write_template(env, "setup.py", setup_args, "setup.py")

    def get_job_dir(self):
        """Returns the GCS path to the job dir."""
        return os.path.join("gs://", self.bucket_id, self.model["name"])

    def get_model_dir(self):
        """Returns the GCS path to the model dir."""
        return os.path.join(self.get_job_dir(), self.model_dir)

    def _get_best_trial(self, job_id):
        """Returns the best trial id for a training job.

        Args:
            job_id: a CAIP job id.

        Returns:
            the trial number that performed the best.
        """
        name = self.get_parent(job=job_id)
        request = self.ml_client.projects().jobs().get(name=name).execute()
        best_trial = "1"
        if "trials" in request["trainingOutput"]:
            best_trial = request["trainingOutput"]["trials"][0]["trialId"]
        return best_trial

    def _get_deployment_dir(self, job_id):
        """Returns the GCS path to the Sklearn exported model.

        Args:
            job_id: a CAIP job id.
        """
        best_trial = self._get_best_trial(job_id)
        output_path = os.path.join(self.get_model_dir(), best_trial)
        return output_path

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

    def _upload_metadata(self, path):
        """Uploads the metadata file necessary to run CAIP explanations.

        Args:
            path: GCS path to the model's *.pb directory
        """
        inputs, outputs = {}, {}
        meta_graph = saved_model_utils.get_meta_graph_def(path, "serve")
        signature_def_key = "serving_default"

        inputs_tensor_info = meta_graph.signature_def[
            signature_def_key].inputs
        outputs_tensor_info = meta_graph.signature_def[
            signature_def_key].outputs

        for feat, input_tensor in sorted(inputs_tensor_info.items()):
            inputs[feat] = {"input_tensor_name": input_tensor.name}

        for label, output_tensor in sorted(outputs_tensor_info.items()):
            outputs[label] = {"output_tensor_name": output_tensor.name}

        explanation_metadata = {
            "inputs": inputs,
            "outputs": outputs,
            "framework": "tensorflow"
        }

        file_name = "explanation_metadata.json"
        with open(file_name, "w+") as output_file:
            json.dump(explanation_metadata, output_file)

        dst = os.path.join(path, file_name)
        tf.io.gfile.copy(file_name, dst, overwrite=True)

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
            request = jobs_client.get(name=self.get_parent(job=job_id))
            response, _ = self._call_ml_client(request)
            state = response["state"]
            print("Job state of {}: {}".format(job_id, state))
        if state != "SUCCEEDED":
            raise RuntimeError(
                "Job didn't succeed. End state: {}".format(state))

    def train(self, tune=False, blocking=True, wait_interval=60):
        """Trains on CAIP.

        Args:
            tune: train with hyperparameter tuning if true.
            blocking: true if the function should exit only once the job
                completes.
            wait_interval: if blocking, how often the job state should be
                checked.

        Returns:
            job_id: a CAIP job id.
        """
        now = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        job_id = "train_{}_{}".format(self.model["name"], now)
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
        if tune:
            hp_config = parse_yaml(self.model_params["hyperparam_config"])
            hyperparams = hp_config["trainingInput"]["hyperparameters"]
            body["trainingInput"]["hyperparameters"] = hyperparams

        request = jobs_client.create(parent=self.get_parent(),
                                     body=body)
        self._call_ml_client(request)
        if blocking:
            self._wait_until_done(job_id, wait_interval)
        return job_id

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
            parent=self.get_parent(), body=body)
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
                name=self.get_parent(operation=op_name))
            response, _ = self._call_ml_client(request)
            done = "done" in response and response["done"]
            print("Operation {} completed: {}".format(op_name, done))

    def _create_version(self, version, job_id, explanations,
                        wait_interval=30):
        """Creates a new version of the model for serving.

        Args:
            version: a version number to use to create a version name.
            job_id: a CAIP job id.
            explanations: whether to create a model that can perform
                CAIP explanations.
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
            "machineType": self.machine_type_pred,
        }
        if explanations:
            exp = self.model_params["explain_output"]
            exp_pm = exp["explain_param"]
            body["explanationConfig"] = {
                exp["explain_type"]: {exp_pm["name"]: exp_pm["value"]}
            }
            self._upload_metadata(self._get_deployment_dir(job_id))
        request = versions_client.create(
            parent=self.get_parent(model=True), body=body)
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
            parent=self.get_parent(model=True))
        response, model_exists = self._call_ml_client(request,
                                                      silent_fail=True)
        return response, model_exists

    @abc.abstractmethod
    def get_deploy_framework(self):
        pass

    def deploy(self, job_id, explanations=False):
        """Deploys model and returns the version name created.

        Args:
            job_id: a CAIP job id.
            explanations: whether to create a model that can perform
                CAIP explanations

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
        return self._create_version(version, job_id, explanations)

    # TODO(humichael): Add option to pass in csv/json file.
    def online_predict(self, inputs, version=""):
        """Uses a deployed model to get predictions for the given inputs.

        Args:
            inputs: a list of feature vectors.
            version: the version name of the deployed model to use. If none is
              provided, the default version will be used.

        Returns:
            a list of predictions.

        Raises:
            RuntimeError: if the deployed model fails to make predictions.
        """
        name = self.get_parent(model=True)
        if version:
            name = self.get_parent(version=version)
        projects_client = self.ml_client.projects()
        request = projects_client.predict(name=name,
                                          body={"instances": inputs})
        response, _ = self._call_ml_client(request)
        if "predictions" in response:
            return response["predictions"]
        print(response)
        raise RuntimeError("Prediction failed.")

    def online_explanations(self, inputs, version=""):
        """Uses a deployed model to get explanations for the given inputs.

        Args:
            inputs: a list of feature vectors.
            version: the version name of the deployed model to use. If none is
                provided, the default version will be used.

        Returns:
            a list of explanations.

        Raises:
            RuntimeError: if the deployed model fails to make explanations.
        """
        name = self.get_parent(model=True)
        if version:
            name = self.get_parent(version=version)
        projects_client = self.ml_client.projects()
        request = projects_client.explain(name=name,
                                          body={"instances": inputs})
        response, _ = self._call_ml_client(request)

        if "explanations" in response:
            return response["explanations"]
        print(response)
        raise RuntimeError("Explanations failed.")

    # TODO(humichael): Move to utils.py
    def upload_pred_input_data(self, src):
        """Uploads input data to GCS for prediction."""
        inputs_dir = os.path.join(self.get_job_dir(), "inputs")
        if not tf.io.gfile.exists(inputs_dir):
            tf.io.gfile.makedirs(inputs_dir)

        src_name = os.path.basename(src)
        dst = os.path.join(inputs_dir, src_name)
        tf.io.gfile.copy(src, dst, overwrite=True)
        return dst

    def get_pred_output_path(self):
        """Returns the path prediction outputs are written to."""
        return os.path.join(self.get_job_dir(), "outputs")

    def supports_batch_predict(self):
        """Returns True if CAIP supports batch prediction for this model."""
        return True

    def batch_predict(self, job_id="", version="", blocking=True,
                      wait_interval=60):
        """Uses a deployed model on GCS to create a prediction job.

        Note: Batch prediction only supports Tensorflow models.

        Args:
            job_id: the job_id of a training job to use for batch prediction.
            version: the version name of the deployed model to use. If none is
              provided, the default version will be used.
            blocking: true if the function should exit only once the job
                completes.
            wait_interval: if blocking, how often the job state should be
                checked.

        Returns:
            job_id: a CAIP job id.

        Raises:
            RuntimeError: if batch prediction is not supported.
        """
        if not self.supports_batch_predict():
            raise RuntimeError("Batch predict not supported for this model.")
        pred_info = self.data["prediction"]
        inputs = pred_info["input_data_paths"]
        if not isinstance(inputs, list):
            inputs = [inputs]
        input_format = (pred_info["input_format"] if "input_format" in pred_info
                        else "DATA_FORMAT_UNSPECIFIED")
        output_format = (pred_info["output_format"]
                         if "output_format" in pred_info else "JSON")
        now = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        predict_id = "predict_{}_{}".format(self.model["name"], now)
        jobs_client = self.ml_client.projects().jobs()
        body = {
            "jobId": predict_id,
            "predictionInput": {
                "dataFormat": input_format,
                "outputDataFormat": output_format,
                "inputPaths": inputs,
                "maxWorkerCount": "10",
                "region": self.region,
                "batchSize": "64",
                "outputPath": self.get_pred_output_path(),
            },
        }
        if job_id:
            body["predictionInput"]["uri"] = self._get_deployment_dir(job_id)
            body["predictionInput"]["runtimeVersion"] = self.runtime_version
        elif version:
            version = self.get_parent(version=version)
            body["predictionInput"]["versionName"] = version
        else:
            model = self.get_parent(model=True)
            body["predictionInput"]["modelName"] = model

        request = jobs_client.create(parent=self.get_parent(),
                                     body=body)
        self._call_ml_client(request)
        if blocking:
            self._wait_until_done(predict_id, wait_interval)
        return predict_id

    # TODO(humichael): clean up with python code, not a shell script.
    def clean_up(self):
        """Delete all generated files."""
        subprocess.call("bin/cleanup.sh")


class SklearnModel(BaseModel):
    """SklearnModel class."""

    def __init__(self, config):
        super(SklearnModel, self).__init__(config, "sklearn")

    def _get_default_input_args(self, train_path, eval_path):
        args = super(SklearnModel, self)._get_default_input_args(
            train_path, eval_path)
        additional_args = {
            "cross_validations": {
                "type": "int",
                "help": "Number of datasets to split to for cross validation.",
                "default": 3,
            },
        }
        args.update(additional_args)
        return args

    def generate_files(self):
        super(SklearnModel, self).generate_files(
            "sklearn_task.py", "sklearn_model.py", "sklearn_inputs.py")

    def get_deploy_framework(self):
        return "SCIKIT_LEARN"

    def supports_batch_predict(self):
        """Returns True if CAIP supports batch prediction for this model."""
        return False


class TFModel(BaseModel):
    """TFModel class."""

    def __init__(self, config):
        super(TFModel, self).__init__(config, "tensorflow")

    def _get_default_input_args(self, train_path, eval_path):
        args = super(TFModel, self)._get_default_input_args(
            train_path, eval_path)
        additional_args = {
            "batch_size": {
                "type": "int",
                "help": "Number of rows of data fed to model each iteration.",
                "default": 64,
            },
            "num_epochs": {
                "type": "int",
                "help": "Number of times to iterate over the dataset.",
            },
            "max_steps": {
                "type": "int",
                "help": "Maximum number of iterations to train the model for.",
                "default": 500,
            },
            "learning_rate": {
                "type": "float",
                "help": "Model learning rate.",
                "default": 0.0001,
            },
            "export_format": {
                "type": "str",
                "help": "File format expected at inference time.",
                "default": "json",
            },
            "save_checkpoints_steps": {
                "type": "int",
                "help": "Steps to run before saving a model checkpoint.",
                "default": 100,
            },
            "keep_checkpoint_max": {
                "type": "int",
                "help": "Number of model checkpoints to keep.",
                "default": 2,
            },
            "log_step_count_steps": {
                "type": "int",
                "help": "Steps to run before logging training performance.",
                "default": 100,
            },
            "eval_steps": {
                "type": "int",
                "help": "Number of steps to use to evaluate the model.",
                "default": 20,
            },
            "early_stopping_steps": {
                "type": "int",
                "help": "Steps with no loss decrease before stopping early.",
                "default": 1000,
            },
        }
        args.update(additional_args)
        return args

    def generate_files(self):
        super(TFModel, self).generate_files(
            "tf_task.py", "tf_model.py", "tf_inputs.py")

    # TODO(humichael): Support multiple model dirs.
    def train(self, tune=False, blocking=True, wait_interval=60):
        """Removes any previous checkpoints before training."""
        if tf.io.gfile.exists(self.get_model_dir()):
            tf.gfile.DeleteRecursively(self.get_model_dir())
        return super(TFModel, self).train(tune, blocking, wait_interval)

    def get_deploy_framework(self):
        return "TENSORFLOW"

    def _get_deployment_dir(self, job_id):
        """Returns the GCS path to the Sklearn exported model.

        Args:
            job_id: a CAIP job id.
        """
        best_trial = self._get_best_trial(job_id)
        output_path = os.path.join(
            self.get_model_dir(), best_trial, "export", "export")
        return str(subprocess.check_output(
            ["gsutil", "ls", output_path]).strip()).split("\\n")[-1].strip("'")


class XGBoostModel(BaseModel):
    """XGBoost class."""

    def __init__(self, config):
        super(XGBoostModel, self).__init__(config, "xgboost")

    def _get_default_input_args(self, train_path, eval_path):
        args = super(XGBoostModel, self)._get_default_input_args(
            train_path, eval_path)
        additional_args = {
            "max_depth": {
                "type": "int",
                "help": "Maximum depth of the XGBoost tree.",
                "default": 3,
            },
            "n_estimators": {
                "type": "int",
                "help": "Number of estimators to be created.",
                "default": 2,
            },
            "booster": {
                "type": "str",
                "help": "which booster to use: gbtree, gblinear or dart.",
                "default": "gbtree",
            },
            "min_child_weight": {
                "type": "int",
                "help": ("Minimum sum of instance weight (hessian) needed in a "
                         "child."),
                "default": 1,
            },
            "learning_rate": {
                "type": "float",
                "help": ("Step size shrinkage used in update to prevents "
                         "overfitting."),
                "default": 0.3,
            },
            "gamma": {
                "type": "int",
                "help": ("Minimum loss reduction required to make a further "
                         "partition on a leaf node of the tree."),
                "default": 0,
            },
            "subsample": {
                "type": "int",
                "help": "Subsample ratio of the training instances.",
                "default": 1,
            },
            "colsample_bytree": {
                "type": "int",
                "help": ("subsample ratio of columns when constructing each "
                         "tree."),
                "default": 1,
            },
            "reg_alpha": {
                "type": "int",
                "help": ("L1 regularization term on weights. Increasing this "
                         "value will make model more conservative."),
                "default": 0,
            },
            "num_classes": {
                "type": "int",
                "help": "Number of output labels must be in [0, num_class).",
                "default": 1,
            },
        }
        args.update(additional_args)
        return args

    def generate_files(self):
        super(XGBoostModel, self).generate_files(
            "xgboost_task.py", "xgboost_model.py", "xgboost_inputs.py")

    def get_deploy_framework(self):
        return "XGBOOST"

    def supports_batch_predict(self):
        """Returns True if CAIP supports batch prediction for this model."""
        return False
