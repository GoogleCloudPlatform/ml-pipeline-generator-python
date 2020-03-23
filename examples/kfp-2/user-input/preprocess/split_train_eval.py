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
"""."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from io import BytesIO
from google.cloud import storage
import argparse


def obtain_train_eval(project_id, bucket_name):
    """."""
    # # All of the data is in a file called Step10_Final_dataset.csv
    print('reading the data file from gcs...')
    print('Project-ID: %s ' %(project_id))
    print('Bucket-ID: %s ' %(bucket_name))


     # The following was derived from the contents of this reply:
     # https://stackoverflow.com/a/50201179
    storage_client = storage.Client(project=project_id, credentials=None)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob('input/Step10_Final_dataset.csv')

    byte_stream = BytesIO()
    blob.download_to_file(byte_stream)
    byte_stream.seek(0)
    df = pd.read_csv(byte_stream)

    # We need to rearrange the columns below just as they shall be
    # expected by the estimator
    print('rearranging data...')
    key_column = 'LOAN_SEQUENCE_NUMBER'
    label_column = 'TARGET'
    bool_cols = []
    int_cols = ['credit_score',
                'mortgage_insurance_percentage',
                'Number_of_units',
                'cltv',
                'original_upb',
                'ltv',
                'original_loan_term',
                'number_of_borrowers',
                'min_CURRENT_DEFERRED_UPB']
    str_cols = ['first_time_home_buyer_flag',
                'occupancy_status',
                'channel',
                'property_state',
                'property_type',
                'loan_purpose',
                'seller_name',
                'service_name']
    # str_nuniques = [2, 3, 3, 52, 5, 2, 20, 24]
    float_cols = ['metropolitan_division',
                  'original_interest_rate',
                  'min_CURRENT_ACTUAL_UPB',
                  'max_CURRENT_ACTUAL_UPB',
                  'Range_CURRENT_ACTUAL_UPB',
                  'stdev_CURRENT_ACTUAL_UPB',
                  'mode_CURRENT_ACTUAL_UPB',
                  'average_CURRENT_ACTUAL_UPB',
                  'max_CURRENT_DEFERRED_UPB',
                  'Range_CURRENT_DEFERRED_UPB',
                  'mode_CURRENT_DEFERRED_UPB',
                  'average_CURRENT_DEFERRED_UPB',
                  'stdev_CURRENT_DEFERRED_UPB',
                  'min_CURRENT_INTEREST_RATE',
                  'max_CURRENT_INTEREST_RATE',
                  'Range_CURRENT_INTEREST_RATE',
                  'mode_CURRENT_INTEREST_RATE',
                  'stdev_CURRENT_INTEREST_RATE',
                  'average_CURRENT_INTEREST_RATE',
                  'PREFINAL_LOAN_DELINQUENCY_STATUS',
                  'frequency_0',
                  'frequency_1',
                  'frequency_2',
                  'frequency_3',
                  'Recency_0',
                  'Recency_1',
                  'Recency_2',
                  'Recency_3']
    # DEFAULTS = [[''] for col in bool_cols] + \
    #            [[0] for col in int_cols] + \
    #            [[0.0] for col in float_cols] + \
    #            [[''] for col in str_cols] + [[''], [0]]
    csv_columns = bool_cols + int_cols + float_cols + \
           str_cols + [key_column, label_column]
    traindata = df[csv_columns]

    # Here, we'll split with a small test size so as to
    # allow our model to train on more data
    print('splitting...')
    x_train, x_test, y_train, y_test = train_test_split(
        traindata.drop(label_column, axis=1), traindata[label_column],
        stratify=traindata[label_column], shuffle=True, test_size=0.1)
    traindf = pd.concat([x_train, y_train], axis=1)
    evaldf = pd.concat([x_test, y_test], axis=1)

    alld = pd.concat([traindf, evaldf])
    strcols = [col for col in alld.columns if alld[col].dtype == 'object']
    if key_column in strcols:
        strcols.remove(key_column)
    alld = pd.get_dummies(alld, columns=strcols)

    divline = traindf.shape[0]
    traindf_wdummies = alld.iloc[:divline, :]
    # not necessary only cmle but can be used to
    # test performance if so desired
    evaldf_wdummies = alld.iloc[divline:, :]
    del alld

    print('Undersample for XG Boost....')

    traindfu_wdummies = pd.concat([
        traindf_wdummies[traindf_wdummies[label_column] == 0].sample(
            frac=0.01),
        traindf_wdummies[traindf_wdummies[label_column] == 1].sample(
            frac=0.55),
        traindf_wdummies[traindf_wdummies[label_column] > 1]])
    traindfu_wdummies = shuffle(traindfu_wdummies)

    # traindfu_wdummies.drop(key_column, axis=1)
    # .to_csv('xgb_train.csv', index=False)
    # evaldf_wdummies.drop([key_column,label_column], axis=1)
    # .to_csv('xgb_eval.csv', index=False)

    # Since the results are small enough to fit in a single
    # well-provisioned VM, we'll write the results to csv files locally
    # then move them to gcs so we have two copies to work
    # with as we please

    print('writing tf model files...')
    write_file(
        storage_client,
        traindf[csv_columns],
        bucket_name,
        'train.csv',
        header=False)
    write_file(
        storage_client,
        evaldf[csv_columns],
        bucket_name,
        'eval.csv',
        header=False)

    # traindf[csv_columns].to_csv('train.csv', index=False, header=False)
    # evaldf[csv_columns].to_csv('eval.csv', index=False, header=False)

    print('writing XG Boost model files...')
    write_file(
        storage_client,
        traindfu_wdummies.drop(key_column, axis=1),
        bucket_name,
        'xgb_train.csv',
        header=True)
    write_file(
        storage_client,
        evaldf_wdummies.drop([key_column, label_column], axis=1),
        bucket_name,
       'xgb_eval.csv',
        header=True)

    with open('./output.txt', 'w') as output_file:
        output_file.write(bucket_name)
        print('Done!')


def write_file(storage_client,
               df,
               bucket_name,
               destination_file_name,
               header):
    """Write a blob from the bucket."""
    df_str = df.to_csv(index=False, header=header)
    # storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob('output/' + destination_file_name)
    blob.upload_from_string(df_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id',
                        type=str,
                        required=True,
                        help='The GCP project_id containing the source file')
    parser.add_argument('--dataset_bucket',
                        type=str,
                        required=True,
                        help='Bucket to store outputs.')
    args = parser.parse_args()

    obtain_train_eval(args.project_id, args.dataset_bucket)
