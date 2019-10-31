from keras import backend as K

K.tensorflow_backend._get_available_gpus()

RESULTS_PATH = "basic_nn_results/"

import dataset_split

all_features_df, all_labels, both_df = dataset_split.feature_label_split()

