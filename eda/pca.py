import numpy as np

import pandas as pd
from collections import Counter

from matplotlib.colors import rgb2hex
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

DATA_PATH = "../data/"
RESULTS_PATH = "pca_results/"

train_df = pd.read_csv(DATA_PATH + 'train.csv')
test_df = pd.read_csv(DATA_PATH + 'test.csv')

train_df['Data'] = 'Train'
test_df['Data'] = 'Test'
both_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
both_df['subject'] = '#' + both_df['subject'].astype(str)

label = both_df.pop('Activity')

def print_data_shape():
    print('Shape Train:\t{}'.format(train_df.shape))
    print('Shape Test:\t{}'.format(test_df.shape))
    print(both_df.info())

    print(train_df.head())

    print(
        pd.DataFrame.from_dict(
        Counter([col.split('-')[0].split('(')[0] for col in both_df.columns]), 
            orient='index')
        .rename(columns={0:'count'})
        .sort_values('count', ascending=False)
    )

    print()

    print('Null Values in DataFrame: {}\n'.format(both_df.isna().sum().sum()))

def plot_distribution():
    label_counts = label.value_counts()
    index = np.arange(len(label_counts))
    counts = label_counts.values

    plt.figure(
        figsize=(18, 5),
        dpi=70)
    plt.title("Distribution")
    plt.bar(index, label_counts.values)
    plt.xticks(index, label_counts.index)
    plt.show()

def activity_pca():
    label_counts = label.value_counts()

    tsne_data = both_df.copy()
    data_data = tsne_data.pop('Data')
    subject_data = tsne_data.pop('subject')

    scl = StandardScaler()
    tsne_data = scl.fit_transform(tsne_data)

    pca = PCA(n_components=0.9, random_state=3)
    tsne_data = pca.fit_transform(tsne_data)

    tsne = TSNE(random_state=3)
    tsne_transformed = tsne.fit_transform(tsne_data)

    fig, axarr = plt.subplots(2, 1, figsize=(15,10))
    n = label.unique().shape[0]
    colormap = get_cmap('viridis')
    colors = [rgb2hex(colormap(col)) for col in np.arange(0, 1.01, 1/(n-1))]

    for i, group in enumerate(label_counts.index):
        # Mask to separate sets
        mask = (label==group).values
        axarr[0].scatter(x=tsne_transformed[mask][:,0], y=tsne_transformed[mask][:,1], c=colors[i], alpha=0.5, label=group)
        axarr[0].set_title('TSNE: Activity Visualisation')
        axarr[0].legend()

    n = subject_data.unique().shape[0]
    colormap = get_cmap('gist_ncar')
    colors = [rgb2hex(colormap(col)) for col in np.arange(0, 1.01, 1/(n-1))]

    for i, group in enumerate(subject_data.unique()):
        # Mask to separate sets
        mask = (subject_data==group).values
        axarr[1].scatter(x=tsne_transformed[mask][:,0], y=tsne_transformed[mask][:,1], c=colors[i], alpha=0.5, label=group)

    axarr[1].set_title('TSNE: Participant Visualisation')
    plt.savefig(RESULTS_PATH + 'ActivitySeperable.pdf')
    plt.show()

activity_pca()
