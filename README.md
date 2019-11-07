# Local Perturbations in a Converged Federated Learning Model

This github repo contains all the models to produce in the results described in the "Local Perturbations in a Converged Federated Learning Model" Master's Thesis.

Our first chosen dataset is the [Human Activity Recognition](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones).
This dataset is also used in [Federated Multi-Task Learning](https://arxiv.org/pdf/1705.10467.pdf). This is found in the HumanActivity folder from the base folder.
We start by performing exploratory data anlysis using PCA and then use [Catboost](https://github.com/catboost/catboost) to create a baseline to see if local models perform better than global models.

## Human Activity Dataset

### Exploratory Data Analysis

PCA Results We also apply T-distributed Stochastic Neighbor Embedding to visualized the results in 2 dimensions. 

<p align="center">
  <img src="https://github.com/JiahanLiu/LocalGlobalFedLearning/blob/master/eda/pca_results/PCA.png">
</p>

Catboost Results

<p align="center">
  <img src="https://github.com/JiahanLiu/LocalGlobalFedLearning/blob/master/eda/catboost_results/Catboost_Results.png">
</p>


