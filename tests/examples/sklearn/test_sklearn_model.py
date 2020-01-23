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
"""Unit tests demo scikit-learn model."""
import unittest

from examples.sklearn import sklearn_model


class TestModel(unittest.TestCase):
    """Tests demo model."""

    def setUp(self):
        super(TestModel, self).setUp()
        self.features, self.labels = sklearn_model.get_data()

    def test_get_data(self):
        """Checks that there is a label for each feature."""
        self.assertEqual(self.features.shape[0], self.labels.shape[0])

    def test_get_model(self):
        """Checks that the model can be trained and used for predictions."""
        model = sklearn_model.get_model()
        model.fit(self.features, self.labels)
        preds = model.predict(self.features)
        self.assertEqual(preds.shape, self.labels.shape)


if __name__ == "__main__":
    unittest.main()
