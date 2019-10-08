import os.path
from os import path

import numpy as np

import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

from catboost import Pool, CatBoostClassifier, cv

import pickle

DATA_PATH = "../data/"
RESULTS_PATH = "results/"

train_df = pd.read_csv(DATA_PATH + 'train.csv')
test_df = pd.read_csv(DATA_PATH + 'test.csv')

both_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
all_features_df = both_df.copy()

all_features_df.pop('subject')
all_labels = all_features_df.pop('Activity')

def cat_boost(model_file, accuracy_file, features_df, labels):
    enc = LabelEncoder()
    label_encoded = enc.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(features_df, label_encoded, test_size=0.2, random_state=3)

    clf = CatBoostClassifier(random_seed=3, 
        custom_loss='TotalF1', 
        loss_function='MultiClass', 
        od_type='Iter', 
        od_wait=20,
        task_type="GPU",
        devices='0',
        )
    if path.exists(model_file):
        clf = pickle.load(open(model_file, 'rb'))
    else:
        clf.fit(X_train, y_train)
        pickle.dump(clf, open(model_file, 'wb'))

    f1_result = f1_score(y_test, clf.predict(X_test), average='macro')
    accuracy_result = accuracy_score(y_test, clf.predict(X_test))
    print("Finished" + model_file)

    with open(accuracy_file, mode='w+') as file: 
        file.write('F1 on testset:\t{:.4f}\n'.format(f1_result))
        file.write('Accuracy on testset:\t{:.4f}\n'.format(accuracy_result))

    print('F1 on testset:\t{:.4f}\n'.format(f1_result))
    print('Accuracy on testset:\t{:.4f}\n'.format(accuracy_result))

def local_catboost():
    model_file_p1 = RESULTS_PATH + 'catboost_'
    model_file_p2 = '.catboost'
    accuracy_file_p1 = RESULTS_PATH + 'CatBoostAccuracy_'
    accuracy_file_p2 = '.txt'

    for i in range(1, 31):
        model_file = model_file_p1 + str(i) + model_file_p2
        accuracy_file = accuracy_file_p1 + str(i) + accuracy_file_p2
        local_df = both_df.loc[both_df['subject'] == i]
        
        local_df.pop('subject')
        local_labels = local_df.pop('Activity')

        cat_boost(model_file, accuracy_file, local_df, local_labels)

def group_catboost5():
    model_file_p1 = RESULTS_PATH + 'catboost_'
    model_file_p2 = '.catboost'
    accuracy_file_p1 = RESULTS_PATH + 'CatBoostAccuracy_'
    accuracy_file_p2 = '.txt'

    for i in range(0, 6):
        lbound = i*5+1
        ubound = i*5+5
        model_file = model_file_p1 + str(lbound) + "_" + str(ubound) + model_file_p2
        accuracy_file = accuracy_file_p1 + str(lbound) + "_" + str(ubound) + accuracy_file_p2
        local_df = both_df.loc[((both_df['subject'] >= lbound) & (both_df['subject'] <= ubound))]

        local_df.pop('subject')
        local_labels = local_df.pop('Activity')

        cat_boost(model_file, accuracy_file, local_df, local_labels)

def catboost_global():
    model_file_p1 = RESULTS_PATH + 'catboost_'
    model_file_p2 = '.catboost'
    accuracy_file_p1 = RESULTS_PATH + 'CatBoostAccuracy_'
    accuracy_file_p2 = '.txt'

    dicts = [
        {
            "model_file": model_file_p1 + "all" + model_file_p2,
            "accuracy_file": accuracy_file_p1 + "all" + accuracy_file_p2,
            "features": all_features_df,
            "labels": all_labels
        }
    ]

    cat_boost(dicts[0]["model_file"], dicts[0]["accuracy_file"], dicts[0]["features"], dicts[0]["labels"])

def catboost_all():
    group_catboost5()
    local_catboost()
    catboost_global()

catboost_all()

