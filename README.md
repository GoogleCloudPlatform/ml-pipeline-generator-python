# ML Pipeline Generator
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ml-pipeline-gen)
[![PyPI version](https://badge.fury.io/py/ml-pipeline-gen.svg)](https://badge.fury.io/py/ml-pipeline-gen)
[![Build
Status](https://travis-ci.com/GoogleCloudPlatform/ml-pipeline-generator-python.svg?branch=master)](https://travis-ci.com/GoogleCloudPlatform/ml-pipeline-generator-python)

ML Pipeline Generator is a tool for generating end-to-end pipelines composed of GCP components so that users can easily migrate their local ML models onto GCP and start realizing the benefits of the Cloud quickly. 

The following ML frameworks will be supported:
1. TensorFlow (TF)
1. Scikit-learn (SKL)
1. XGBoost (XGB)

The following backends are currently supported for model training:
1. [Google Cloud AI Platform](https://cloud.google.com/ai-platform) 
1. [AI Platform Pipelines](https://cloud.google.com/ai-platform/pipelines/docs) (managed Kubeflow Pipelines)

## Installation
```bash
pip install ml-pipeline-gen
```

## Setup
### GCP credentials
```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project [PROJECT_ID]
```

### Enabling required APIs

The tool requires following Google Cloud APIs to be enabled: 
1. [Compute Engine](https://console.cloud.google.com/apis/api/compute.googleapis.com)
1. [AI Platform Training and Prediction](https://console.cloud.google.com/apis/api/ml.googleapis.com)
1. [Cloud Storage](https://console.cloud.google.com/apis/api/storage-component.googleapis.com)

Enable the above APIs by following the links, or run the below command to enable the APIs for your project.

```bash
gcloud services enable ml.googleapis.com \
compute.googleapis.com \
storage-component.googleapis.com
```

### Python environment
```bash
python3 -m venv venv
source ./venv/bin/activate
pip install ml-pipeline-gen
```

### Kubeflow
Create a Kubeflow deployment using Cloud Marketplace. Follow these
[instructions](https://github.com/kubeflow/pipelines/blob/master/manifests/gcp_marketplace/guide.md#gcp-service-account-credentials)
to give the Kubeflow instance access to GCP services.

> A future release will automate provisioning of KFP clusters and incorporate
K8s Workload Identity for auth. 

## Cloud AI Platform Demo
This demo uses the scikit-learn model in
`examples/sklearn/model/sklearn_model.py` to create a training module to run on
CAIP. First, make a copy of the scikit-learn example.

```bash
cp -r examples/sklearn sklearn-demo
cd sklearn-demo
```

Create a `config.yaml` by using the `config.yaml.example` template. See the
[Input args](#input-args) section for details on the config parameters. Once the
config file is filled out, run the demo.

```bash
python demo.py
```

Running this demo uses the config file to generate a `trainer/` module that is
compatible with CAIP.

## KFP Demo
This demo uses the TensorFlow model in `examples/kfp/model/tf_model.py` to
create a KubeFlow Pipeline (hosted on AI Platform Pipelines). First, make a copy
of the kfp example.

```bash
cp -r examples/kfp kfp-demo
cd kfp-demo
```

Create a `config.yaml` by using the `config.yaml.example` template. See the
[Input args](#input-args) section for details on the config parameters. Once the
config file is filled out, run the demo.

```bash
python demo.py
```

Running this demo uses the config file to generate a `trainer/` module that is
compatible with CAIP. It also generates `orchestration/pipeline.py`, which
compiles a Kubeflow Pipeline.

## Tests
The tests use `unittest`, Python's built-in unit testing framework. By running
`python -m unittest`, the framework performs test discovery to find all tests
within this project. Tests can be run on a more granular level by feeding a
directory to test discover. Read more about `unittest`
[here](https://docs.python.org/3/library/unittest.html).

```bash
python -m unittest
```
## Input args
The following input args are included by default. Overwrite them by adding them
as inputs in the config file.

| Arg | Description |
| ------------- | ----- |
| train_path| Dir or bucket containing train data.|
| eval_path | Dir or bucket containing eval data.|
| model_dir | Dir or bucket to save model files. |
| batch_size | Number of rows of data to be fed into the model each iteration. |
| max_steps | The maximum number of iterations to train the model for. |
| learning_rate| Multiplier that controls how much the weights of our network are adjusted with respoect to the loss gradient.|
| export_format | File format expected by the exported model at inference time. |
| save_checkpoints_steps | Number of steps to run before saving a model checkpoint. |
| keep_checkpoint_max | Number of model checkpoints to keep. |
| log_step_count_steps | Number of steps to run before logging training performance. |
| eval_steps | Number of steps to use to evaluate the model. |
| early_stopping_steps | Number of steps with no loss decrease before stopping early. |

## Contribute
To modify the behavior of the library, install `ml-pipeline-gen` using:

```bash
pip install -e ".[dev]"
```
