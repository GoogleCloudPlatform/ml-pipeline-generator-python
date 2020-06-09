### hptune_config.yaml schema

Below schema should be used when preparing a `hptune_config.yaml` file for models using the tool. The parameters follow the Cloud AI Platform [HyperparameterSpec](https://cloud.google.com/ai-platform/training/docs/reference/rest/v1/projects.jobs#HyperparameterSpec), some of which are optional and marked as such.

<pre>
<b>trainingInput</b>:
	<b>hyperparameters</b>:
		<b>goal</b>: [the type of goal to use for tuning, MAXIMIZE or MINIMIZE]
		<b>params</b>: [the set of parameters to tune]
			- <b>parameterName</b>: [unique parameter name, e.g. “learning_rate”]
			  <b>type</b>: [parameter <a href="https://cloud.google.com/ai-platform/training/docs/reference/rest/v1/projects.jobs#ParameterType">type</a>]
			  <b>minValue</b>: [min value for the parameter, if DOUBLE or INTEGER type]
			  <b>maxValue</b>: [max value for the parameter, if DOUBLE or INTEGER type]
			  <b>scaleType</b>: [optional; how the parameter should be <a href="https://cloud.google.com/ai-platform/training/docs/reference/rest/v1/projects.jobs#ScaleType">scaled</a>]
		<b>maxTrials</b>: [optional; how many training trials should be attempted to optimize the specified hyperparameters]
		<b>maxParallelTrials</b>: [optional; the number of training trials to run concurrently]
		<b>maxFailedTrials</b>: [optional; the number of failed trials that need to be seen before failing the hyperparameter tuning job]
		<b>hyperparameterMetricTag</b>: [optional; TensorFlow summary tag name to use for optimizing trials]
		<b>resumePreviousJobId</b>: [optional; the prior hyperparameter tuning job id that users hope to continue with]
		<b>enableTrialEarlyStopping</b>: [optional; indicates if the hyperparameter tuning job enables auto trial early stopping]
		<b>algorithm</b>: [optional; search <a href="https://cloud.google.com/ai-platform/training/docs/reference/rest/v1/projects.jobs#Algorithm">algorithm</a> to be used]
</pre>
