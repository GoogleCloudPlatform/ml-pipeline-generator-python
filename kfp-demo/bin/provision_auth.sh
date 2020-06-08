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

# check_scope=$(gcloud container clusters describe $CLUSTER_NAME | grep $SCOPES)
# if [[ $check_scope != *"$SCOPES"* ]]; then
#   echo "ERROR: The KFP cluster does not not have cloud-platform scope. Check 
#   https://cloud.google.com/ai-platform/pipelines/docs/setting-up#full-access for
#   more details."
#   exit 1
# fi

# gcloud container clusters update $CLUSTER_NAME --enable-autoprovisioning \
#   --zone $ZONE \
#   --autoprovisioning-scopes $SCOPES

# check_scope=$(gcloud container clusters describe cluster-6 | grep 'cloud-platform')

# if [[ $check_scope != *"cloud-platform"* ]]; then
#   echo "WARNING: The KFP cluster may not have right access scopes."
# fi


# gcloud services enable ml.googleapis.com \
#   compute.googleapis.com \
#   container.googleapis.com \
#   containerregistry.googleapis.com

gcloud container clusters update $CLUSTER_NAME \
  --zone $ZONE \
  --workload-pool "${PROJECT_ID}".svc.id.goog

gcloud container node-pools update default-pool \
  --cluster $CLUSTER_NAME \
  --workload-metadata GKE_METADATA

gcloud container clusters get-credentials $CLUSTER_NAME
gcloud iam service-accounts create $gsa_name

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member serviceAccount:"${gsa_name}"@"${PROJECT_ID}".iam.gserviceaccount.com \
  --role "roles/editor"

kubectl create namespace $namespace
kubectl create serviceaccount --namespace $namespace $ksa_name

gcloud iam service-accounts add-iam-policy-binding \
  --role roles/iam.workloadIdentityUser \
  --member "serviceAccount:${PROJECT_ID}.svc.id.goog[${namespace}/${ksa_name}]" \
  "${gsa_name}"@"${PROJECT_ID}".iam.gserviceaccount.com

kubectl annotate serviceaccount \
  --namespace $namespace \
  $ksa_name \
  iam.gke.io/gcp-service-account="${gsa_name}"@"${PROJECT_ID}".iam.gserviceaccount.com

# kubectl run -it \
#   --generator=run-pod/v1 \
#   --image google/cloud-sdk:slim \
#   --serviceaccount $ksa_name \
#   --namespace $namespace \
#   workload-identity-test


# gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
#   --member serviceAccount:"${CLUSTER_NAME}"-kfp-user@"${PROJECT_ID}".iam.gserviceaccount.com \
#   --role roles/storage.admin