# AI Pipelines
AI Pipelines is a tool for generating end-to-end pipelines composed of GCP components so that any customer can easily migrate their local ML models onto GCP and start realizing the benefits of the cloud quickly. Currently ML pipelines are very difficult to implement for customers, and take weeks if not months with experienced Googlers.

The following ML frameworks will be supported:
1. Tensorflow (TF)
1. Scikit-learn (SKL)
1. XGBoost (XGB)

We will first only consider Kubeflow Pipelines (KFP) for orchestrating ML pipelines built using various Cloud AI Platform (CAIP) features. Orchestration using Cloud Composer (CC) may be in scope in the future.

The full project plan can be found [here](https://docs.google.com/document/d/11-ljj4D3UT-_bOyFeN_L_uRXQkM9G10bte9jy1yfSYA/edit?ts=5df59215).
