#!/bin/bash

# Copyright 2019 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

while [ $# -ne 0 ]; do
    case "$1" in
       -h|--help)      echo "Usage: ./hptune.sh \\"
                       echo "--region=<REGION> \\"
                       echo "--module-name=<MODULE_NAME> \\"
                       echo "--package-path=<PACKAGE_PATH> \\"
                       echo "--job-dir=<JOB_DIR> \\"
                       echo "--staging-bucket=<STAGING_BUCKET> \\"
                       echo "--config=<CONFIG> \\"
                       echo "--runtime-version=<RUNTIME_VERSION> \\"
                       echo "--stream-logs \\"
                       echo "-- \\"
                       echo "--common_args=<COMMON/NON_TUNABLE_ARGS>"
                       exit
		       shift
                       ;;
       --region)       REGION=$2
                       shift
                       ;;
       --python_module)  MODULE_NAME=$2
                       shift
                       ;;
       --package_uri) PACKAGE_URI=$2
                       shift
                       ;;
       --job_dir)      JOB_DIR=$2
                       shift
                       ;;
      --staging_bucket)STAGING_BUCKET=$2
                       shift
                       ;;
       --config)       CONFIG=$2
                       shift
                       ;;
     --runtime_version)RUNTIME_VERSION=$2
                       shift
                       ;;
       --args)   ARGS=$2
                       shift
                       ;;
       ###	
       --job_id) JOB_ID=$2      
                       shift
                       ;;
       *)              shift
                        ;;
    esac
done   
echo "Executing $0 $@ . ...."
COMMON_ARGS=`python -c "import ast; print(' '.join(ast.literal_eval('$ARGS')))"`
COMMON_ARGS=`echo $COMMON_ARGS |  sed 's/--\([^ ]*\) *\([^-]*\)/--\1=\2/g'`

JOBNAME=wd_hcr_hptuning_$(date -u +%y%m%d_%H%M)

gsutil -m rm -rf $OUTPUT_DIR  || echo "No object was deleted" 
gsutil -m cp $CONFIG .
config_file=`basename $CONFIG`

eval `echo "gcloud ai-platform jobs submit training $JOBNAME \
       --region=$REGION \
       --module-name=$MODULE_NAME \
       --packages=$PACKAGE_URI \
       --job-dir=$JOB_DIR \
       --staging-bucket=$STAGING_BUCKET \
       --config=$config_file \
       --runtime-version=$RUNTIME_VERSION \
       --stream-logs \
       -- \
       $COMMON_ARGS

"`

mkdir -p `dirname $JOB_ID`

echo "$JOBNAME" > $JOB_ID
