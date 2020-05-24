#!/bin/bash
#
# Copyright 2020 Google LLC
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
#
# Script to set up Google service accounts and workload identity bindings for a Kubeflow Pipelines (KFP) standalone deployment.
# Adapted for AI Platforms from https://github.com/kubeflow/pipelines/blob/master/manifests/kustomize/gcp-workload-identity-setup.sh
#
# What the script configures:
#      1. Google service accounts (GSAs): $SYSTEM_GSA and $USER_GSA.
#      2. Service account IAM policy bindings.
#      3. Kubernetes service account annotations.
# 
# Requirements:
#      1. gcloud set up in the environment calling the script
#      2. KFP is deployed on a GKE cluster
#      3. kubectl is set to talk to the GKE cluster
set -e

# TODO(ashokpatelapk): Lint this shell script.
# Google service Account (GSA)
SYSTEM_GSA=${SYSTEM_GSA:-$CLUSTER_NAME-kfp-system}
USER_GSA=${USER_GSA:-$CLUSTER_NAME-kfp-user}

# Kubernetes Service Account (KSA)
SYSTEM_KSA=(ml-pipeline-ui ml-pipeline-visualizationserver)
USER_KSA=(pipeline-runner default)

# Fetch the namespace env var, and set it to 'kubeflow' if no env var is set
NAMESPACE=${NAMESPACE:-kubeflow}

if [ -z "$PROJECT_ID" ]; then
  echo "Error: PROJECT_ID env variable is empty!"
  exit 1
fi

if [ -z "$CLUSTER_NAME" ]; then
  echo "Error: CLUSTER_NAME env variable is empty!"
  exit 1
fi

echo "Env variables set:"
echo "* PROJECT_ID=$PROJECT_ID"
echo "* CLUSTER_NAME=$CLUSTER_NAME"
echo "* NAMESPACE=$NAMESPACE"

echo "Creating Google service accounts..."
function create_gsa_if_not_present {
  local name=${1}
  local already_present=$(gcloud iam service-accounts list --filter='name:'$name'' --format='value(name)')
  if [ -n "$already_present" ]; then
    echo "Service account $name already exists"
  else
    gcloud iam service-accounts create $name
  fi
}

create_gsa_if_not_present $SYSTEM_GSA
create_gsa_if_not_present $USER_GSA

# Add iam policy bindings to grant project permissions to these GSAs.
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SYSTEM_GSA@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/editor"
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$USER_GSA@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/editor"

# Bind KSA to GSA through workload identity.
function bind_gsa_and_ksa {
  local gsa=${1}
  local ksa=${2}

  gcloud iam service-accounts add-iam-policy-binding $gsa@$PROJECT_ID.iam.gserviceaccount.com \
    --member="serviceAccount:$PROJECT_ID.svc.id.goog[$NAMESPACE/$ksa]" \
    --role="roles/iam.workloadIdentityUser" \
    > /dev/null

  kubectl annotate serviceaccount \
    --namespace $NAMESPACE \
    --overwrite \
    $ksa iam.gke.io/gcp-service-account=$gsa@$PROJECT_ID.iam.gserviceaccount.com

  echo "* Bound KSA $ksa to GSA $gsa"
}

echo "Binding each kfp system KSA to $SYSTEM_GSA"
for ksa in ${SYSTEM_KSA[@]}; do
  bind_gsa_and_ksa $SYSTEM_GSA $ksa
done

echo "Binding each kfp user KSA to $USER_GSA"
for ksa in ${USER_KSA[@]}; do
  bind_gsa_and_ksa $USER_GSA $ksa
done
