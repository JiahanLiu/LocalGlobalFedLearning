import pandas as pd

from sklearn.preprocessing import LabelEncoder

DATA_PATH = "../data/"

def feature_label_split():
    train_df = pd.read_csv(DATA_PATH + 'train.csv')
    test_df = pd.read_csv(DATA_PATH + 'test.csv')

    both_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    all_features_df = both_df.copy()

    all_features_df.pop('subject')
    all_labels = all_features_df.pop('Activity')

    return all_features_df, all_labels, both_df