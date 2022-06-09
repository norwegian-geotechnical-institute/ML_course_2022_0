"""Script for analysis of rock type classification model"""

import torch
from pathlib import Path
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# load trained model and do some predictions
##############################################################################
# model = torch.load("Reports/rock_model.pth")

# load training data and analyse
#############################################################################
perf = pickle.load(open("Reports/rock_classification_performance.pkl", "rb"))
trues = perf[-1]["test_labels"]
preds = perf[-1]["test_predictions"]

plt.rcParams.update({'font.size': 6})
cm = confusion_matrix(trues, preds, normalize="true")
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(15,15))
disp.plot(cmap="viridis", ax=ax, values_format=".2f")
ax.set_xlabel("Predicted rocktype")
ax.set_ylabel("True rocktype")
plt.tight_layout()
plt.savefig("Figures/confusion_matrix_rocktypes.png", dpi=600)
# plt.show()
