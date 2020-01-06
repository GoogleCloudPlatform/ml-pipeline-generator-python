#!/bin/bash
#
# Copyright 2019 Google Inc. All Rights Reserved.
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
#
# Convenience script for training model on AI Platform.
TYPE=${1:-"local"}

NOW=$(date +"%Y%m%d_%H%M%S")
PROJECT_ID="{{project_id}}"
BUCKET_ID="{{bucket_id}}"
JOB_NAME=demo_${NOW}
JOB_DIR=gs://$BUCKET_ID/demo
PACKAGE_PATH=trainer
MODULE_NAME=trainer.task
REGION=us-central1
RUNTIME_VERSION=1.15
PYTHON_VERSION=3.5
SCALE_TIER=BASIC
MODEL_DIR="gs://${BUCKET_ID}/demo/models"
  
if [ "${TYPE}" == "cloud" ]; then
  gcloud ai-platform jobs submit training "${JOB_NAME}" \
    --job-dir "${JOB_DIR}" \
    --package-path "${PACKAGE_PATH}" \
    --module-name "${MODULE_NAME}" \
    --region "${REGION}" \
    --runtime-version=${RUNTIME_VERSION} \
    --python-version=${PYTHON_VERSION} \
    --scale-tier "${SCALE_TIER}" \
    -- \
    --model_dir "${MODEL_DIR}"
else
  gcloud ai-platform local train \
    --package-path "${PACKAGE_PATH}" \
    --module-name "${MODULE_NAME}" \
    --
fi
