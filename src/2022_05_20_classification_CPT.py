"""
NGI internal Machine Learning course.
Second Session 13.05.2022
Topic: Machine learning classification of CPT-data

@author: Tom F. Hansen, Georg H. Erharter
VSCode version: 1.67.1
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    ConfusionMatrixDisplay,
    balanced_accuracy_score)
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE

# READ IN DATA
# ***********************************************************************
X_train = np.load(Path('../Data/processed/CPT_X_train.npy'))
X_test = np.load(Path('../Data/processed/CPT_X_test.npy'))
y_train = np.load(Path('../Data/processed/CPT_y_train.npy')).flatten() # flatten
y_test = np.load(Path('../Data/processed/CPT_y_test.npy')).flatten() # flatten

# SCALE DATA
# ***********************************************************************
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# BALANCE DATA - Oversampling to equal number of samples in each class
# ***********************************************************************
sm = SMOTE(random_state=42)
# print(np.unique(y_train, return_counts=True))
X_train, y_train = sm.fit_resample(X_train, y_train)
# print(np.unique(y_train, return_counts=True))

# DEFINES CLASSIFIER AND FIT MODEL TO DATA
# ***********************************************************************
# clf = DummyClassifier() # starts with a dummy classifier
# print(np.unique(y_test, return_counts=True)[1]/len(y_test))
clf = KNeighborsClassifier(n_jobs=-1, n_neighbors=5)
clf.fit(X_train, y_train)

#PREDICT
# ***********************************************************************
y_test_predict = clf.predict(X_test)

#REPORT RESULTS
# ***********************************************************************
print(classification_report(y_test, y_test_predict, zero_division=0))
print(f"Bal. acc: {balanced_accuracy_score(y_test, y_test_predict): .2f}") # one good metric for comparison

cm = confusion_matrix(y_test, y_test_predict, normalize="true")
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(1,9))
fig, ax = plt.subplots(figsize=(15,10))
disp.plot(cmap="Greys", ax=ax, values_format=".2f")
ax.set_xlabel("Predicted soil class")
ax.set_ylabel("True soil class")
plt.show()

