"""
NGI internal Machine Learning course.
Session 27.05.2022
Topic: Machine learning classification of CPT-data

@author: Tom F. Hansen, Georg H. Erharter
"""
# presenting better error messages
from rich.traceback import install
install()

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    ConfusionMatrixDisplay,
    balanced_accuracy_score
    )
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from utility import DebugPipeline
# models
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import lightgbm as lgbm

# READ IN DATA
# ***********************************************************************
ROOT = Path.cwd() # current working directory
X_train = np.load(Path(ROOT,"Data/processed/CPT_X_train.npy"))
X_test = np.load(Path(ROOT,"Data/processed/CPT_X_test.npy"))
y_train = np.load(Path(ROOT,"Data/processed/CPT_y_train.npy")).flatten() # flatten
y_test = np.load(Path(ROOT,"Data/processed/CPT_y_test.npy")).flatten() # flatten

np.random.seed(42) #to stop randomization, for comparing ML-algorithms

# Defining a pipeline for processing and classification in same process
clf = Pipeline(steps=[
    ("scaler", MinMaxScaler()),
    ("debug", DebugPipeline(plot=False, pca=False)),
    # ("downscaling", PCA(n_components=2)),
    # ("undersampling", RandomUnderSampler(
        # sampling_strategy={0:8165, 1:17359,  2:6563,  3:2028, 4:20000, 5:20000, 
                        #    6:20000, 7:20000})),
    ("balancing", SMOTE()),
    ("classifier", KNeighborsClassifier(n_jobs=-1)),
    # ("classifier", RandomForestClassifier(n_jobs=-1)),
    # ("classifier", LogisticRegression()),
    # ("classifier", lgbm.LGBMClassifier()),
    # ("classifier", MLPClassifier(hidden_layer_sizes=(20, 10))),
])

clf.fit(X_train, y_train)

#PREDICT
# ***********************************************************************
y_test_predict = clf.predict(X_test)

#REPORT RESULTS
# ***********************************************************************
print(classification_report(y_test, y_test_predict, zero_division=0))
print(f"Bal. acc: {balanced_accuracy_score(y_test, y_test_predict): .3f}") # one good metric for comparison

cm = confusion_matrix(y_test, y_test_predict, normalize="true")
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(1,9))
fig, ax = plt.subplots(figsize=(15,10))
disp.plot(cmap="Greys", ax=ax, values_format=".2f")
ax.set_xlabel("Predicted soil class")
ax.set_ylabel("True soil class")
plt.show()


# CROSS VALIDATION - GIVING MORE REPRESENTATIVE METRICS
# ***********************************************************************
X = np.concatenate((X_train, X_test),axis=0)
y = np.concatenate((y_train, y_test), axis=0)
scores = cross_val_score(clf, X, y, cv=5, scoring="balanced_accuracy", n_jobs=-1)
print(f"Mean value: {scores.mean():.3f}")
print(f"Standard deviation: {scores.std():.3f}")
