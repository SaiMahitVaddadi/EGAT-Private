from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

class KNNClassifier:
    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto'):
        """
        Initialize the K-NN classifier.

        Parameters:
        - n_neighbors: Number of neighbors to use (default: 5)
        - weights: Weight function used in prediction (default: 'uniform')
        - algorithm: Algorithm used to compute the nearest neighbors (default: 'auto')
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights, algorithm=self.algorithm)

    def train(self, train_x, train_y):
        """
        Train the K-NN model.

        Parameters:
        - train_x: Training features (2D array-like)
        - train_y: Training labels (1D array-like)
        """
        self.model.fit(train_x, train_y)

    def predict(self, test_x):
        """
        Predict using the trained K-NN model.

        Parameters:
        - test_x: Test features (2D array-like)

        Returns:
        - predictions: Predicted labels (1D array-like)
        """
        return self.model.predict(test_x)

class KNNRegressor:
    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto'):
        """
        Initialize the K-NN regressor.

        Parameters:
        - n_neighbors: Number of neighbors to use (default: 5)
        - weights: Weight function used in prediction (default: 'uniform')
        - algorithm: Algorithm used to compute the nearest neighbors (default: 'auto')
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.model = KNeighborsRegressor(n_neighbors=self.n_neighbors, weights=self.weights, algorithm=self.algorithm)

    def train(self, train_x, train_y):
        """
        Train the K-NN regressor.

        Parameters:
        - train_x: Training features (2D array-like)
        - train_y: Training target values (1D array-like)
        """
        self.model.fit(train_x, train_y)

    def predict(self, test_x):
        """
        Predict using the trained K-NN regressor.

        Parameters:
        - test_x: Test features (2D array-like)

        Returns:
        - predictions: Predicted target values (1D array-like)
        """
        return self.model.predict(test_x)
