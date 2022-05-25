# -*- coding: utf-8 -*-
"""
NGI internal Machine Learning course.
Second Session 13.05.2022
Topic: Data Preprocessing for ML

Created on Wed May 11 11:20:16 2022
@author: Georg H. Erharter, Tom F. Hansen

CPT data script

setup:
    Anaconda Python distribution
    https://www.anaconda.com/products/distribution
    python==3.9.7
    matplotlib==3.5.2
    numpy==1.19.5
    pandas==1.4.1
    scikit-learn==1.0.2
    seaborn==0.11.2
    Integrated Development Environment (IDE)
    spyder==5.1.5
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
from sklearn.model_selection import train_test_split

###############################################################################
# static variables

FILEPATH = Path('../Data/raw/CPT_PremstallerGeotechnik_revised.csv')
INPUT_FEATURES = ['Qtn (-)', 'Fr (%)', 'U2 (-)']  # input features for ML
OUTPUT = ['Oberhollenzer_classes']
TRAIN_TEST_SPLIT = 0.25  # fraction of the data for testing

###############################################################################
# "normal" data preprocessing

# use pandas to read the file
df = pd.read_csv(FILEPATH)

# get basic information about the dataframe
print(df)
print(df.info(show_counts=True))  # get info about datatype and NaN
# get statistics on columns of dataframe and safe to a new excel sheet
df.describe().to_excel(Path('../Data/processed/basic_statistics.xlsx'))

# drop all datapoints where there are no labels
df.dropna(subset=INPUT_FEATURES+OUTPUT, inplace=True)

print(f'n datapoints before outlier removal: {len(df)}')
# manually delete outliers
df = df[df['Qtn (-)'] <= 1000]
df = df[df['Qtn (-)'] >= 1]

df = df[df['Fr (%)'] <= 10]
df = df[df['Fr (%)'] >= 0.1]

df = df[df['U2 (-)'] <= 20]
df = df[df['U2 (-)'] >= -2]
print(f'n datapoints after outlier removal: {len(df)}')

# make histograms to look at data distributions
fig = plt.figure(figsize=(4*len(INPUT_FEATURES), 4))
for i, feature in enumerate(INPUT_FEATURES):
    ax = fig.add_subplot(1, len(INPUT_FEATURES), i+1)
    ax.hist(df[feature], bins=30, edgecolor='black')
    ax.set_xlabel(feature)
plt.tight_layout()
plt.savefig(Path("../Figures/feature_histograms.png"))

# sns.pairplot(df[FEATURES])
# Feature engineering if required

###############################################################################
# "ML" data preprocessing

# take logarithm of exponentially distributed features
for feature in ['Qtn (-)', 'Fr (%)']:
    df[feature] = np.log(df[feature])

# split into input and output data
X = df[INPUT_FEATURES].values
y = df[OUTPUT].values.astype(int)  # classification labels

# split into training and testing data
# stratified sample
print(np.unique(y, return_counts=True)[1]/len(y)*100)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=TRAIN_TEST_SPLIT,
                                                    random_state=42,
                                                    stratify=y)
print(np.unique(y_train, return_counts=True)[1]/len(y_train)*100)

# # normalize / scale features between 0 and 1 according to training data
# # get min values of training input and subtract from data
# min_vals = X_train.min(axis=0)
# X_train = X_train - min_vals
# X_test = X_test - min_vals
# # get new max values of training input and divide data by it
# max_vals = X_train.max(axis=0)
# X_train = X_train / max_vals
# X_test = X_test / max_vals

# print(X_train.min(axis=0), X_train.max(axis=0))
# print(X_test.min(axis=0), X_test.max(axis=0))

# # save data to files
# np.save(Path('Data/processed/CPT_min_vals.npy', min_vals))
# np.save(Path('Data/processed/CPT_max_vals.npy', max_vals))

np.save(Path('../Data/processed/CPT_X_train.npy'), X_train)
np.save(Path('../Data/processed/CPT_y_train.npy'), y_train)
np.save(Path('../Data/processed/CPT_X_test.npy'), X_test)
np.save(Path('../Data/processed/CPT_y_test.npy'), y_test)

