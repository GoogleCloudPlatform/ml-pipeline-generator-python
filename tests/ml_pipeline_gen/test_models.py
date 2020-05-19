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
import os
import unittest

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
    def setUpClass(cls):
        """Instantiates a model."""
        super(TestSklearnModel, cls).setUpClass()
        cls.config = "examples/sklearn/config.yaml"
        cls.model = SklearnModel(cls.config)

    @classmethod
    def tearDownClass(cls):
        """Cleans up generated directories."""
        super(TestSklearnModel, cls).tearDownClass()
        cls.model.clean_up()

    # TODO(humichael): technically private functions don't need to be tested. It
    # should reflect in public functions.
    def test_set_config(self):
        """Ensures instance variables are created."""
        model = self.__class__.model
        model.model = {}

        model._set_config(self.__class__.config)
        self.assertEqual(model.model["name"], "sklearn_demo_model")

    def test_populate_trainer(self):
        """Ensures task.py and model.py are created."""
        model = self.__class__.model
        model.clean_up()

        model._populate_trainer()
        trainer_files = os.listdir("trainer")
        self.assertIn("task.py", trainer_files)
        self.assertIn("model.py", trainer_files)

    @unittest.skip("How to test without running training?")
    def test_local_train(self):
        """Tests local training."""
        model = self.__class__.model
        model.train()
        model_files = os.listdir("models")
        self.assertIn("{}.joblib".format(model.model["name"]), model_files)

    # TODO(humichael): Need to spoof CAIP calls to test this.
    def test_cloud_train(self):
        """Tests training on CAIP."""
        pass

    # TODO(humichael): Need to spoof CAIP calls to test this.
    def test_serve(self):
        """Tests serving."""
        pass


if __name__ == "__main__":
    unittest.main()
