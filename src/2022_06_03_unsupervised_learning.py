# -*- coding: utf-8 -*-
"""
NGI internal Machine Learning course.
Session 03.06.2022
Topic: Applying unsupervised Machine Learning to the CPT dataset

Dataset:
https://www.tugraz.at/institute/ibg/forschung/numerische-geotechnik/datenbank/
Paper to dataset:
https://doi.org/10.1016/j.dib.2020.106618

@author: Georg H. Erharter, Tom F. Hansen
"""

# presenting better error messages
from rich.traceback import install
install()


import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.cluster import DBSCAN, KMeans, MeanShift
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from umap import UMAP

###############################################################################
# fixed variables and constants

FEATURES = ['Qtn (-)', 'Fr (%)', 'U2 (-)', 'qc (MPa)', 'fs (kPa)', 'u2 (kPa)',
            'qt (MPa)', 'Rf (%)', 'Qt (-)']
SOIL_CLASSES = ['no class', "Gr,sa,si' - Gr,co", "Or,cl - Or,sa'",
                'Or,sa - Or/Sa', 'Sa,gr,si - Gr,sa,si', "Sa,si - Sa,gr,si'",
                "Si,sa,cl' - Si,sa,gr", "Cl/Si,sa' - Si,cl,sa"]
N_DATAPOINTS = 10_000


###############################################################################
# data import and preprocessing

np.random.seed(42)

ROOT = Path.cwd()
FILEPATH = Path(ROOT, 'Data/raw/CPT_PremstallerGeotechnik_revised.csv')

df = pd.read_csv(FILEPATH, low_memory=False)

# removing outliers
df = df[df['Qtn (-)'] <= 1000]
df = df[df['Qtn (-)'] >= 1]

df = df[df['Fr (%)'] <= 10]
df = df[df['Fr (%)'] >= 0.1]

df = df[df['U2 (-)'] <= 20]
df = df[df['U2 (-)'] >= -2]


# take logarithm of exponentially distributed features
for feature in ['Qtn (-)', 'Fr (%)', 'qc (MPa)', 'fs (kPa)', 'u2 (kPa)',
                'qt (MPa)', 'Rf (%)', 'Qt (-)']:
    df[feature] = np.log(df[feature])

# delete infinite (due to log) and nan values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=FEATURES, inplace=True)

X = df[FEATURES].values  # convert to numpy array

# get random sample of whole dataset
ids = np.random.choice(np.arange(len(X)), size=N_DATAPOINTS)
# data to process
X = X[ids]
# soil labels for verification
soil_classes = df['Oberhollenzer_classes'].iloc[ids]
soil_classes.fillna('nan', inplace=True)

# scale data between 0 and 1
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

###############################################################################
# dimensionality reduction

reducer_1 = PCA(n_components=2)
reducer_2 = TSNE(perplexity=150, learning_rate='auto', init='pca', n_jobs=-1)
reducer_3 = UMAP(n_jobs=-1)

for reducer, name in zip([reducer_1, reducer_2, reducer_3],
                         ['PCA', 't-SNE', 'UMAP']):
    X_transformed = reducer.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 8))
    for sc in ['nan', 0, 1, 2, 3, 4, 5, 6, 7]:
        soil_id = np.where(soil_classes == sc)[0]
        if sc == 'nan':
            ax.scatter(X_transformed[:, 0][soil_id], X_transformed[:, 1][soil_id],
                       color='white', edgecolor='black', alpha=0.5, label=sc)
        else:
            ax.scatter(X_transformed[:, 0][soil_id], X_transformed[:, 1][soil_id],
                       edgecolor='black', alpha=1, s=60,
                       label=SOIL_CLASSES[sc])
    ax.set_title(name)
    ax.legend()
    plt.tight_layout()
    plt.show()

###############################################################################
# clustering

clusterer_1 = KMeans(n_clusters=7)  # look for 7 clusters
clusterer_2 = MeanShift(n_jobs=-1)  # finds its own clusters
clusterer_3 = DBSCAN(eps=0.16, n_jobs=-1)  # finds its own clusters

for clusterer, name in zip([clusterer_1, clusterer_2, clusterer_3],
                           ['KMeans', 'MeanShift', 'DBSCAN']):
    new_classes = clusterer.fit_predict(X_transformed)
    print(f'{name} found {len(np.unique(new_classes))} clusters')

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(X_transformed[:, 0], X_transformed[:, 1], c=new_classes,
               cmap='tab10')

    ax.set_title(name)
    plt.tight_layout()
    plt.show()
