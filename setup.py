# Copyright 2020 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Config for installing a Python module/package."""

import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="ml-pipeline-gen",
    version="0.0.3",
    author="Michael Hu",
    author_email="author@example.com",
    description="A tool for generating end-to-end pipelines on GCP.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GoogleCloudPlatform/ml-pipeline-generator-python",
    packages=["ml_pipeline_gen"],
    install_requires=[
        "cloudml-hypertune",
        "gcsfs",
        "google-api-python-client",
        "jinja2",
        "joblib",
        "kfp",
        "pandas",
        "pyyaml",
        "scikit-learn",
        "tensorflow>=0.14.0,<2.0.0",
        "xgboost",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    include_package_data=True,
)
