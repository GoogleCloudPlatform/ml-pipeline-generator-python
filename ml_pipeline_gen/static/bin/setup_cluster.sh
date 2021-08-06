#!/bin/bash
#
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
#
# Script to create a Kubeflow Pipelines cluster and configure it with Workload
# Identity to allow the K8s cluster to access other GCP resources such as GCS.
usage() {
  echo "Usage: ./setup_cluster.sh -n GKE_CLUSTER_NAME -z GKE_CLUSTER_ZONE [-m MACHINE_TYPE] [-v KFP_VERSION]"
  exit 1
}

PIPELINE_VERSION=0.5.1
MACHINE_TYPE="n1-standard-2"

while getopts ":n:z:m:v:" opts; do
  case "${opts}" in
    n)
      CLUSTER_NAME=${OPTARG}
      ;;
    z)
      ZONE=${OPTARG}
      ;;
    m)
      MACHINE_TYPE=${OPTARG}
      ;;
    v)
      PIPELINE_VERSION=${OPTARG}
      ;;
    *)
      usage
      ;;
  esac
done

if [ -z "${CLUSTER_NAME}" ] || [ -z "${ZONE}" ]; then
    usage
fi

PROJECT_ID=$(gcloud config get-value project)
NAMESPACE="kubeflow"
SCOPES=cloud-platform

# Google service Account (GSA)
SYSTEM_GSA=$CLUSTER_NAME-kfp-system
USER_GSA=$CLUSTER_NAME-kfp-user

# Kubernetes Service Account (KSA)
SYSTEM_KSA=(ml-pipeline-ui ml-pipeline-visualizationserver)
USER_KSA=(pipeline-runner default)

gcloud config set project $PROJECT_ID
gcloud config set compute/zone $ZONE

gcloud services enable ml.googleapis.com \
  compute.googleapis.com \
  container.googleapis.com \
  containerregistry.googleapis.com

gcloud beta container clusters create "${CLUSTER_NAME}" \
  --zone=$ZONE \
  --machine-type=$MACHINE_TYPE \
  --disk-type="pd-standard" \
  --disk-size="100" \
  --num-nodes="3" \
  --scopes=$SCOPES  \
  --identity-namespace="${PROJECT_ID}".svc.id.goog

gcloud container clusters get-credentials $CLUSTER_NAME \
  --zone=$ZONE

kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=${PIPELINE_VERSION}"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=${PIPELINE_VERSION}"

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

# Bind and annotate KSAs.
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

echo "KFP cluster created and bindings set up."

gcloud container clusters update $CLUSTER_NAME \
  --zone=$ZONE \
  --update-labels mlpg_wi_auth=true

sleep 30

echo "The KFP Dashboard hostname is:"
kubectl describe configmap inverse-proxy-config -n kubeflow | grep googleusercontent.com

echo "Please update the config.yaml with the cluster details and KFP hostname before deploying your models."
