from sklearn.svm import SVC
from sklearn.svm import SVR

class SVMClassifier:
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        """
        Initialize the SVM classifier.

        Parameters:
        - kernel: Kernel type to be used in the algorithm (default: 'rbf')
        - C: Regularization parameter (default: 1.0)
        - gamma: Kernel coefficient for 'rbf', 'poly', and 'sigmoid' (default: 'scale')
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.model = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma)

    def train(self, train_x, train_y):
        """
        Train the SVM model.

        Parameters:
        - train_x: Training features (2D array-like)
        - train_y: Training labels (1D array-like)
        """
        self.model.fit(train_x, train_y)

    def predict(self, test_x):
        """
        Predict using the trained SVM model.

        Parameters:
        - test_x: Test features (2D array-like)

        Returns:
        - predictions: Predicted labels (1D array-like)
        """
        return self.model.predict(test_x)
class SVMRegressor:
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        """
        Initialize the SVM regressor.

        Parameters:
        - kernel: Kernel type to be used in the algorithm (default: 'rbf')
        - C: Regularization parameter (default: 1.0)
        - gamma: Kernel coefficient for 'rbf', 'poly', and 'sigmoid' (default: 'scale')
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.model = SVR(kernel=self.kernel, C=self.C, gamma=self.gamma)

    def train(self, train_x, train_y):
        """
        Train the SVM regressor.

        Parameters:
        - train_x: Training features (2D array-like)
        - train_y: Training target values (1D array-like)
        """
        self.model.fit(train_x, train_y)

    def predict(self, test_x):
        """
        Predict using the trained SVM regressor.

        Parameters:
        - test_x: Test features (2D array-like)

        Returns:
        - predictions: Predicted target values (1D array-like)
        """
        return self.model.predict(test_x)
