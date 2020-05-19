# ML Pipeline Generator
ML Pipeline Generator is a tool for generating end-to-end pipelines composed of GCP components so that users can easily migrate their local ML models onto GCP and start realizing the benefits of the Cloud quickly. 

The following ML frameworks will be supported:
1. TensorFlow (TF)
1. Scikit-learn (SKL)
1. XGBoost (XGB)

The following backends are currently supported for model training:
1. [Google Cloud AI Platform](https://cloud.google.com/ai-platform) 
1. [AI Platform Pipelines](https://cloud.google.com/ai-platform/pipelines/docs) (managed Kubeflow Pipelines)

## Setup
### GCP credentials
```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project [PROJECT_ID]
```

### Python environment
```bash
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

### Config file
Update the information in `config.yaml`. See the [Input args](#input-args) section below for details on the config parameters. 

### Kubeflow
Create a Kubeflow deployment using Cloud Marketplace. Follow these
[instructions](https://github.com/kubeflow/pipelines/blob/master/manifests/gcp_marketplace/guide.md#gcp-service-account-credentials)
to give the Kubeflow instance access to GCP services.

> A future release will automate provisioning of KFP clusters and incorporate K8s Workload Identity for auth. 

## Cloud AI Platform Demo
This demo uses the scikit-learn model in `examples/sklearn/user_model.py` to
create a training module to run on CAIP.

```bash
python -m examples.sklearn.demo
```

Running this demo uses the config file to generate `bin/run.train.sh` along
with `trainer/` code. Then, run `bin/run.train.sh` to train locally or
`bin/run.train.sh cloud` to train on Google Cloud AI Platform.

## KFP Demo
This demo uses the scikit-learn model in `examples/sklearn/user_model.py` to
create a KubeFlow Pipeline (hosted on AI Platform Pipelines).

```bash
python -m examples.kfp.demo
python -m orchestration.pipeline
```

### Cleanup
Delete the generated files by running `bin/cleanup.sh`.

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
