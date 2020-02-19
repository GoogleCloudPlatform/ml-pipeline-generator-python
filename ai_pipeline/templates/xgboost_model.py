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
"""ML model definitions."""
import logging
import hypertune
import os

import pandas as pd
import numpy as np

from trainer import utils
from sklearn import metrics
from sklearn import preprocessing
from xgboost import XGBClassifier

TARGET_COLUMN = 'TARGET'


def train_and_evaluate(args, output_dir):
    # load-into-pandas
    traindf = pd.read_csv(args.TRAIN_FILE)
    X_train, y_train = traindf.drop(TARGET_COLUMN, axis=1), traindf[
        TARGET_COLUMN]
    logging.info("Read training file")

    # count number of classes
    values, counts = np.unique(y_train, return_counts=True)
    NUM_CLASSES = len(values)

    # ---------------------------------------
    # Train model
    # ---------------------------------------

    params = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'booster': args.booster,
        'min_child_weight': args.min_child_weight,
        'learning_rate': args.learning_rate,
        'gamma': args.gamma,
        'subsample': args.subsample,
        'colsample_bytree': args.colsample_bytree,
        'reg_alpha': args.reg_alpha,
        'num_class': NUM_CLASSES
    }
    logging.info("Starting to train...")
    xgb_model = XGBClassifier(**params)
    xgb_model.fit(X_train, y_train)
    logging.info("Completed training XGBOOST model")

    bst = xgb_model.get_booster()
    bst_filename = 'model.bst'
    bst.save_model(bst_filename)
    model_output_path = os.path.join(output_dir, "model",
                                     bst_filename)
    utils.upload_blob(model_output_path.split("/")[2], bst_filename,
                      "/".join(model_output_path.split("/")[3:]))

    logging.info("Successfully uploaded file to GCS at location %s",
                 args.output_dir)

    # load-into-pandas
    testdf = pd.read_csv(args.TEST_FILE)
    X_test, y_test = testdf.drop(TARGET_COLUMN, axis=1), testdf[
        TARGET_COLUMN]
    logging.info("Read test file")

    # predict the model with test file
    y_pred = xgb_model.predict(X_test)

    # Binarize multiclass labels
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    # Define the score we want to use to evaluate the classifier on
    score = metrics.roc_auc_score(y_test, y_pred, average='macro')
    logging.info("AUC Score: %s", str(score))

    # The default name of the metric is training/hptuning/metric.
    # We recommend that you assign a custom name. The only functional difference is that
    # if you use a custom name, you must set the hyperparameterMetricTag value in the
    # HyperparameterSpec object in your job request to match your chosen name.
    # https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#HyperparameterSpec
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='roc_auc',
        metric_value=score,
        global_step=1000
    )
