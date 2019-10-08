from keras import backend as K

K.tensorflow_backend._get_available_gpus()

DATA_PATH = "../data/"
RESULTS_PATH = "basic_nn_results/"

train_df = pd.read_csv(DATA_PATH + 'train.csv')
test_df = pd.read_csv(DATA_PATH + 'test.csv')

