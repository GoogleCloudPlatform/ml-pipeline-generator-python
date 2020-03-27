# python3
# Copyright 2019 Google Inc. All Rights Reserved.
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
# python3
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
"""Train a simple SVM classifier."""

import argparse
import numpy as np

from sklearn import metrics
from xgboost import XGBClassifier

from examples.preprocess.taxi_preprocess import load_data


TARGET_COLUMN = "TARGET"

def get_model(args):
    """Returns a XGBoost model."""
    params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "booster": args.booster,
        "min_child_weight": args.min_child_weight,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "reg_alpha": args.reg_alpha,
        "num_class": args.num_classes
    }
    xgb_model = XGBClassifier(**params)
    return xgb_model


def main():
    """Trains a model locally to test get_model()."""
    train_x, train_y, eval_x, eval_y = load_data()
    train_y, eval_y = [np.ravel(x) for x in [train_y, eval_y]]
    params = argparse.Namespace(
        n_estimators = 2,
        max_depth = 3,
        booster = "gbtree",
        min_child_weight = 1,
        learning_rate = 0.3,
        gamma = 0,
        subsample = 1,
        colsample_bytree = 1,
        reg_alpha = 0,
        num_class = 1)
    model = get_model(params)
    model.fit(train_x, train_y)
    y_pred = model.predict(eval_x)
    score = metrics.roc_auc_score(eval_y, y_pred, average="macro")
    print("ROC: {}".format(score))


if __name__ == "__main__":
    main()
