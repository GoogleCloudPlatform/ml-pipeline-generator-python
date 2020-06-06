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
# Script to create a Kubeflow Pipelines cluster and configure it for Workload
# Identities to allow the K8s cluster to access other GCP resources such as GCS
# in the project
export PROJECT_ID=$1
export CLUSTER_NAME=$2
export ZONE=$3
export SCOPES=cloud-platform

gcloud config set project "${PROJECT_ID}"
gcloud config set compute/zone "${ZONE}"

gcloud services enable ml.googleapis.com \
  compute.googleapis.com \
  container.googleapis.com \
  containerregistry.googleapis.com

gcloud container clusters update $CLUSTER_NAME \
  --workload-pool="${PROJECT_ID}".svc.id.goog

gcloud container node-pools update default-pool \
  --cluster=$CLUSTER_NAME \
  --workload-metadata=GKE_METADATA

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member serviceAccount:"${CLUSTER_NAME}"-kfp-user@"${PROJECT_ID}".iam.gserviceaccount.com \
  --role roles/storage.admin
