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
# Script to set up Google service accounts and workload identity bindings for a 
# Kubeflow Pipelines (KFP) standalone deployment.
#
# The script checks if the GKE cluster has Workload Identity enabled and 
# configured with a custom label, and if not, enables it and updates the label.
# 
# Adapted for ML Pipeline Generator from https://github.com/kubeflow/pipelines/blob/master/manifests/kustomize/gcp-workload-identity-setup.sh
#
# What the script configures:
#      1. Workload Identity for the cluster.
#      2. Google service accounts (GSAs): $SYSTEM_GSA and $USER_GSA.
#      3. Service account IAM policy bindings.
#      4. Kubernetes service account annotations.
#
# Note: Since the node pool is updated with WI, a new KFP hostname is generated.
# 
# Requirements:
#      1. gcloud set up in the environment calling the script
#      2. KFP is deployed on a GKE cluster 
set -e

# Cluster vars
PROJECT_ID=$1
CLUSTER_NAME=$2
ZONE=$3
NAMESPACE=$4

echo "Workload Identity has not been provisioned for "${CLUSTER_NAME}" ("${ZONE}"), enabling it now..."

# Google service Account (GSA)
SYSTEM_GSA=$CLUSTER_NAME-kfp-system
USER_GSA=$CLUSTER_NAME-kfp-user

# Kubernetes Service Account (KSA)
SYSTEM_KSA=(ml-pipeline-ui ml-pipeline-visualizationserver)
USER_KSA=(pipeline-runner default)

gcloud container clusters get-credentials $CLUSTER_NAME \
  --zone=$ZONE

gcloud container clusters update $CLUSTER_NAME \
  --zone=$ZONE \
  --workload-pool="${PROJECT_ID}".svc.id.goog 

gcloud beta container node-pools update default-pool \
  --cluster=$CLUSTER_NAME \
  --zone=$ZONE \
  --max-surge-upgrade=3 \  
  --max-unavailable-upgrade=0

gcloud container node-pools update default-pool \
  --cluster=$CLUSTER_NAME \
  --zone=$ZONE \
  --workload-metadata=GKE_METADATA

echo "Creating Google Service Accounts..."
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

gcloud container clusters update $CLUSTER_NAME \
  --zone=$ZONE
  --update-labels mlpg_wi_auth=true

RED='\033[0;31m'
COLOR_RESET='\033[0m'
echo -e "${RED}Workload Identity has been enabled, and KFP dashboard URL has been updated. Please update the hostname in config.yaml for future runs.${COLOR_RESET}"
