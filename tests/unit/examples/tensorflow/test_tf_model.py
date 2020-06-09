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
"""Unit tests demo TF model."""
import argparse
import os
import shutil
import sys
import tempfile
import unittest

import tensorflow.compat.v1 as tf

from tests import test_utils


class TestModel(tf.test.TestCase):
    """Tests TF demo model."""

    @classmethod
    def setUpClass(cls):
        super(TestModel, cls).setUpClass()
        cls.test_dir = tempfile.mkdtemp()
        cls.demo_dir = os.path.join(cls.test_dir, 'demo')
        shutil.copytree('examples/tf', cls.demo_dir)

        # TODO(humichael) We can't import the model using __import__ because
        # several other examples are also adding their demo dirs to sys.path.
        # It's very likely the model module that is imported is not the one from
        # this test. All examples currently use the same census_preprocess.
        # These tests will break if any example uses a different preprocessing
        # script.
        # We should just mock this.
        sys.path.append(cls.demo_dir)
        tf_model = test_utils.load_module(
            'tf_model', os.path.join(cls.demo_dir, 'model', 'tf_model.py'))
        tf_preprocess = test_utils.load_module(
            'tf_preprocess', os.path.join(
                cls.demo_dir, 'model', 'census_preprocess.py'))
        sys.path.remove(cls.demo_dir)

        cls.features, cls.labels, _, _ = tf_preprocess.load_data()
        cls.model = tf_model

    @classmethod
    def tearDownClass(cls):
        super(TestModel, cls).tearDownClass()
        shutil.rmtree(cls.test_dir)

    # pylint: disable=g-import-not-at-top
    def setUp(self):
        super(TestModel, self).setUp()
        self.model = self.__class__.model
        self.features = self.__class__.features
        self.labels = self.__class__.labels

    def test_get_data(self):
        """Checks that there is a label for each feature."""
        self.assertEqual(self.features.shape[0], self.labels.shape[0])

    def test_get_model(self):
        """Checks that the model can be trained and used for predictions."""
        input_layer = tf.keras.layers.Input(shape=(self.features.shape[1],))
        params = argparse.Namespace(first_layer_size=50, num_layers=5)
        predictions = self.model.get_model(input_layer, params)

        model = tf.keras.models.Model(inputs=input_layer, outputs=predictions)
        model.compile(optimizer='adam', loss=tf.losses.sigmoid_cross_entropy,
                      metrics=['accuracy'])
        model.fit(self.features, self.labels)
        preds = model.predict(self.features)
        self.assertEqual(preds.shape[0], self.labels.shape[0])


if __name__ == '__main__':
    unittest.main()
