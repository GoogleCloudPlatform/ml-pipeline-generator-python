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
"""Unit tests for models classes."""
import mock
import os
import shutil
import tempfile
import unittest

from googleapiclient import discovery

from ml_pipeline_gen.models import BaseModel
from ml_pipeline_gen.models import SklearnModel


class TestBaseModel(unittest.TestCase):
    """Tests BaseModel class."""

    def test_init(self):
        """Ensure BaseModel is abstract."""
        with self.assertRaises(TypeError):
            BaseModel()


class TestSklearnModel(unittest.TestCase):
    """Tests SklearnModel class."""

    @classmethod
    @mock.patch.object(discovery, 'build')
    def setUpClass(cls, build_mock):
        """Copies a demo and instantiates a model."""
        super(TestSklearnModel, cls).setUpClass()
        build_mock.return_value = None
        cls.cwd = os.getcwd()
        cls.test_dir = tempfile.mkdtemp()
        cls.demo_dir = os.path.join(cls.test_dir, 'demo')
        shutil.copytree('examples/sklearn', cls.demo_dir)

        os.chdir(cls.demo_dir)
        cls.config = 'config.yaml.example'
        cls.model = SklearnModel(cls.config)

    @classmethod
    def tearDownClass(cls):
        """Switch back to the original working dir and removes the demo."""
        super(TestSklearnModel, cls).tearDownClass()
        os.chdir(cls.cwd)
        shutil.rmtree(cls.test_dir)

    def setUp(self):
        super(TestSklearnModel, self).setUp()
        self.model = self.__class__.model

    def tearDown(self):
        super(TestSklearnModel, self).tearDown()
        try:
            self.__class__.model.clean_up()
        except FileNotFoundError:
            pass

    def test_generate_files(self):
        """Ensures task.py and model.py are created."""
        self.assertFalse(os.path.exists('trainer'))
        self.model.generate_files()
        self.assertTrue(os.path.exists('trainer'))
        trainer_files = os.listdir('trainer')
        self.assertIn('task.py', trainer_files)
        self.assertIn('model.py', trainer_files)

    @unittest.skip('How to test without running training?')
    def test_local_train(self):
        """Tests local training."""
        self.model.generate_files()
        self.model.train()
        model_files = os.listdir('models')
        self.assertIn('{}.joblib'.format(self.model.model['name']), model_files)

    # TODO(humichael): Need to spoof CAIP calls to test this.
    def test_cloud_train(self):
        """Tests training on CAIP."""
        pass

    # TODO(humichael): Need to spoof CAIP calls to test this.
    def test_serve(self):
        """Tests serving."""
        pass


if __name__ == '__main__':
    unittest.main()
