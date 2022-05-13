# -*- coding: utf-8 -*-
"""
NGI internal Machine Learning course.
Second Session 13.05.2022
Topic: Data Preprocessing for ML

Created on Wed May 11 11:20:16 2022
@author: Georg H. Erharter, Tom F. Hansen

Rock data script

setup:
    Anaconda Python distribution
    https://www.anaconda.com/products/distribution
    python==3.9.7
    matplotlib==3.5.1
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
import seaborn as sns
from sklearn.model_selection import train_test_split


###############################################################################
# function definitions

def min_max_scaling(data, min_val=None, max_val=None):
    '''custom function for scaling data between 0 and 1'''
    if min_val is None:
        min_val = data.min(axis=0)
    data = data - min_val
    if max_val is None:
        max_val = data.max(axis=0)
    data = data / max_val
    return data, min_val, max_val

###############################################################################
# static variables

FILEPATH = r'Data\Geomechanical_data_ZENODO.xlsx'
FEATURES = ['ultrasonic_Vp_m_per_s', 'ultrasonic_Vs_m_per_s']  # input features
TRAIN_TEST_SPLIT = 0.25  # fraction of the data for testing

###############################################################################
# "normal" data preprocessing

# use pandas to read the file
df = pd.read_excel(FILEPATH)

# get basic information about the dataframe
print(df)
print(df.info(show_counts=True))  # get info about datatype and NaN
# get statistics on columns of dataframe and safe to a new excel sheet
df.describe().to_excel('basic_statistics.xlsx')

# drop all datapoints where there are no labels
df.dropna(subset=['ultrasonic_Vp_m_per_s', 'ultrasonic_Vs_m_per_s', 'UCS_MPa'],
          inplace=True)

# make histograms to look at data distributions
fig = plt.figure(figsize=(4*len(FEATURES), 4))
for i, feature in enumerate(FEATURES):
    ax = fig.add_subplot(1, len(FEATURES), i+1)
    ax.hist(df[feature], bins=30, edgecolor='black')
    ax.set_xlabel(feature)
plt.tight_layout()

sns.pairplot(df[FEATURES+['UCS_MPa']])
# Feature engineering if required

###############################################################################
# "ML" data preprocessing

# split into input and output data
X = df[FEATURES].values
y = df['UCS_MPa'].values  # regression targets

# split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=TRAIN_TEST_SPLIT,
                                                    random_state=42)

# normalize / scale features between 0 and 1 according to training data for
# both input und output data, since this is a regression problem
X_train, min_val_X, max_val_X = min_max_scaling(X_train)
X_test = min_max_scaling(X_test, min_val_X, max_val_X)[0]

y_train, min_val_y, max_val_y = min_max_scaling(y_train)
y_test = min_max_scaling(y_test, min_val_y, max_val_y)[0]

print(X_train.min(axis=0), X_train.max(axis=0))
print(X_test.min(axis=0), X_test.max(axis=0))

# save data to files
np.save(r'Data\Rock_X_train.npy', X_train)
np.save(r'Data\Rock_y_train.npy', y_train)
np.save(r'Data\Rock_X_test.npy', X_test)
np.save(r'Data\Rock_y_test.npy', y_test)

np.save(r'Data\Rock_min_val_X.npy', min_val_X)
np.save(r'Data\Rock_max_val_X.npy', max_val_X)
np.save(r'Data\Rock_min_val_y.npy', min_val_y)
np.save(r'Data\Rock_max_val_y.npy', max_val_y)
