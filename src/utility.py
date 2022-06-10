"""
NGI internal Machine Learning course.

Utility class for other scripts.

@author: Tom F. Hansen, Georg H. Erharter
"""
from pip import main
from sklearn.base import TransformerMixin, BaseEstimator
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy.typing as npt
import numpy as np


class DebugPipeline(BaseEstimator, TransformerMixin):
    """Class to investigate features and labels in a Scikit-learn Pipeline.
    
    Args:
    - plot: Setting plot to True will plot a scatter of the 2 first features
    - pca:  Setting pca to True will plot a scatter using the 2 first pca components
    """
    def __init__(self, plot:bool=False, pca:bool=False)->None:
        self.plot = plot
        self.pca = pca

    def transform(self, X: npt.NDArray)->npt.NDArray:
        """Transforming incoming features.
        Method is invoked upon call by fit and predict on pipeline object.
        Will first show data about X_train and then X_test"""
        
        print(f"Shape of dataset-partition: {X.shape}")
        print("First rows of dataset:")
        print(X[0:5]);print()
        X_vals = X.copy()
        if self.plot:
            self.plot_features(X_vals)
        # setting state variables for easy retrival of data
        self.shape = X.shape
        self.X = X
        # include other interesting variables
        return X

    def fit(self, X, y=None, **fit_params):
        return self
    
    def plot_features(self, X: npt.NDArray)->None:
        """Plot feature information"""
        fig, ax = plt.subplots()
        if self.pca:
            # print 2 PCA components of scatter
            pca = PCA(n_components=2)
            X = pca.fit_transform(X)
        ax.scatter(X[:,0], X[:,1], s=5)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        plt.show()
        
        
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    
        
def examplify():
    """Examplifies functionality"""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.decomposition import PCA
    
    features = np.arange(24).reshape((6,4))
    pipe = Pipeline(steps=[
        ("scaler", MinMaxScaler()),
        ("debug", DebugPipeline(plot=True, pca=False)),
        ("downsamling", PCA(n_components=2))
    ])
    features_trans = pipe.fit_transform(features)
    print(f"After transformation\n{features_trans}")
    # to access values in state-variables directly
    print(pipe["debug"].X)

if __name__=="__main__":
    examplify()