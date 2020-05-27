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
"""Functions for parsing data sources."""
import types
import yaml


# TODO(humichael): Replace with gfile to support GCS.
def parse_yaml(path):
    """Parses the given config file."""
    with open(path, "r") as f:
        doc = f.read()
    return yaml.load(doc, Loader=yaml.FullLoader)


class NestedNamespace(types.SimpleNamespace):
    """Parse nested disctionary to create nested namespace object."""

    def __init__(self, dictionary, **kwargs):
        super(NestedNamespace, self).__init__(**kwargs)
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.__setattr__(key, NestedNamespace(value))
            elif isinstance(value, list):
                self.__setattr__(key,
                                 [NestedNamespace(i)
                                  if isinstance(i, dict)
                                  else i for i in value])
            else:
                self.__setattr__(key, value)

