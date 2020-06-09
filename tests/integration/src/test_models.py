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
"""Integration tests for models classes."""
import mock
import os
import shutil
import tempfile
import time
import unittest

from googleapiclient import discovery
from tensorflow.io import gfile

from ml_pipeline_gen.models import BaseModel
from ml_pipeline_gen.models import SklearnModel


class TestSklearnModel(unittest.TestCase):
    """Tests SklearnModel class."""

    @classmethod
    def setUpClass(cls):
        """Copies a demo and instantiates a model."""
        super(TestSklearnModel, cls).setUpClass()
        cls.cwd = os.getcwd()
        cls.test_dir = tempfile.mkdtemp()
        cls.demo_dir = os.path.join(cls.test_dir, 'demo')
        shutil.copytree('examples/sklearn', cls.demo_dir)
        shutil.copyfile('tests/integration/fixtures/test_config.yaml',
                        os.path.join(cls.demo_dir, 'test_config.yaml'))
        os.chdir(cls.demo_dir)

    @classmethod
    def tearDownClass(cls):
        """Switch back to the original working dir and removes the demo."""
        super(TestSklearnModel, cls).tearDownClass()
        os.chdir(cls.cwd)
        shutil.rmtree(cls.test_dir)

    def modify_config(self):
        self.model.model['name'] = 'test_model_{}'.format(self.now)
        self.model.model['path'] = 'model.sklearn_model'
        self.model.model_params['input_args']['C'] = {
            'type': 'float',
            'default': 1.0,
        }

    def setUp(self):
        super(TestSklearnModel, self).setUp()
        # Delete models if exists
        self.now = int(time.time())
        self.model = SklearnModel('test_config.yaml')
        self.modify_config()

        self.gcs_path = 'gs://ml-pipeline-gen-test/test_model_{}'.format(
            self.now)
        self.model_dir = os.path.join(self.gcs_path, 'models')

    def tearDown(self):
        super(TestSklearnModel, self).tearDown()
        self.model.clean_up()
        if gfile.exists(self.gcs_path):
            gfile.rmtree(self.gcs_path)

    def test_cloud_train(self):
        """Tests training on CAIP."""
        self.model.generate_files()
        self.model.train(tune=False)

        self.assertTrue(gfile.exists(self.model_dir))
        export_path = os.path.join(self.model_dir, '1', 'model.joblib')
        self.assertTrue(gfile.exists(export_path))


if __name__ == '__main__':
    unittest.main()
