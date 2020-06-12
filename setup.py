# Copyright 2020 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Config for installing a Python module/package."""

import setuptools
import ml_pipeline_gen

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='ml-pipeline-gen',
    version=ml_pipeline_gen.__version__,
    author='Michael Hu',
    author_email='author@example.com',
    description='A tool for generating end-to-end pipelines on GCP.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/GoogleCloudPlatform/ml-pipeline-generator-python',
    packages=['ml_pipeline_gen'],
    install_requires=[
        'cloudml-hypertune>=0.1.0',
        'gcsfs>=0.6.2',
        'google-api-python-client>=1.9.3',
        'google-cloud-container>=0.5.0',
        'jinja2>=2.11.2',
        'joblib>=0.15.1',
        'kfp>=0.5.1',
        'pandas>=1.0.4',
        'pyyaml>=5.3.1',
        'scikit-learn>=0.23.1',
        'tensorflow>=1.14.0,<2.0.0',
        'xgboost>=1.1.1',
    ],
    extras_require={
        'dev': [
            'mock',
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    include_package_data=True,
)
