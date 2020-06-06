
### config.yaml schema

Below schema should be used when preparing a config.yaml file for models using the tool. Some parameters are optional and marked as such.

<pre><code><b>project_id</b>: [project ID]
<b>bucket_id</b>: [GCS bucket ID]
<b>region</b>: [GCP region to train ML Pipeline Generator models in, on AI Platform]
<b>scale_tier</b>: [compute <a href="https://cloud.google.com/ai-platform/training/docs/machine-types#scale_tiers">specifications</a> for training the model on AI Platform]
<b>runtime_version</b>: [AI Platform Training <a href="https://cloud.google.com/ai-platform/training/docs/runtime-version-list">runtime version</a>]
<b>python_version</b>: [Python version used in the model code for training]
<b>package_name</b>: [name for the source distribution to be uploaded to GCS]
<b>machine_type_pred</b>: [type of <a href="https://cloud.google.com/ai-platform/training/docs/runtime-version-list">virtual machine</a> that AI Platform Prediction uses for the nodes that serve predictions, defaults to mls1-c1-m2]

<b>data</b>:
	<b>schema</b>:
		- [schema for input & target features in the training data]
	<b>train</b>: [GCS location url to upload preprocessed training data]
	<b>evaluation</b>: [GCS location url to upload preprocessed eval data]
	<b>prediction</b>:
		<b>input_data_paths</b>:
			- [GCS location urls for prediction input data]
		<b>input_format</b>: [prediction input format]
		<b>output_format</b>: [prediction output format]

<b>model</b>:
	<b>name</b>: [unique model name, must start with a letter and only contain letters, numbers, and underscores]
	<b>path</b>: [local dir path to the model.py file]
	<b>target</b>: [target feature in training data]
	<b>metrics</b>: [metrics to evaluate model training on, such as “accuracy”]

<b>model_params</b>:
	<b>input_args</b>: [Any input params to be submitted with the job]
		<b>arg_name</b>:
			<b>type</b>: [data type of the arg, such as int]
			<b>help</b>: [short description of the arg]
			<b>default</b>: [default value of the arg]
	<b>hyperparam_config</b>: [optional; local path to <a href="https://cloud.google.com/ai-platform/training/docs/reference/rest/v1/projects.jobs#hyperparameterspec">hyperparam tuning</a> config yaml. See schema <a href="HPTUNE_CONFIG.md">here</a> for this config file.]
	<b>explanation</b>: [optional; <a href="https://cloud.google.com/ai-platform/training/docs/reference/rest/v1/projects.models.versions#explanationconfig">explainability features</a> for the training job]

<b>orchestration</b>:
	<b>kubeflow_url</b>: [for KFP backend; URL of preconfigured Kubeflow instance]<code></pre>