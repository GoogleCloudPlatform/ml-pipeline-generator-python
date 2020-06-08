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

from setuptools import find_packages
from setuptools import setup

NAME = "cchatterjee-mlpg"
VERSION = "1.0"
REQUIRED_PACKAGES = ["gcsfs"]

setup(
    name=NAME,
    version=VERSION,
    author="Author",
    author_email="author@example.com",
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    url="www.example.com",
)