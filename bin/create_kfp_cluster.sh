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
export PIPELINE_VERSION=0.2.5

gcloud config set project "${PROJECT_ID}"
gcloud config set compute/zone "${ZONE}"

gcloud services enable ml.googleapis.com \
  compute.googleapis.com \
  container.googleapis.com \
  containerregistry.googleapis.com

if (( $(dpkg-query -W -f="${Status}" kubectl 2>/dev/null | grep -c "ok installed") != 0 ));
then
  sudo apt-get install kubectl
fi

# TODO(ashokpatelapk): Handle case when cluster already exists i.e. set up WI for existing cluster
gcloud beta container clusters create "${CLUSTER_NAME}" \
  --zone "${ZONE}" \
  --machine-type "n1-standard-1" \
  --disk-type "pd-standard" \
  --disk-size "100" \
  --num-nodes "3" \
  --scopes "${SCOPES}"  \
  --identity-namespace="${PROJECT_ID}".svc.id.goog

gcloud container clusters get-credentials "${CLUSTER_NAME}"

kubectl apply -k github.com/kubeflow/pipelines/manifests/kustomize/base/crds?ref="${PIPELINE_VERSION}"

kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io

kubectl apply -k github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref="${PIPELINE_VERSION}"

./bin/workload_identity_setup.sh

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member serviceAccount:"${CLUSTER_NAME}"-kfp-user@"${PROJECT_ID}".iam.gserviceaccount.com \
  --role roles/storage.admin

echo "KFP cluster created and bindings set up."
