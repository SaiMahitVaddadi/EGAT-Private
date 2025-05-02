import numpy as np
from scipy.spatial.distance import cdist
from numpy.linalg import inv

class Kriging:
    def __init__(self, variogram_model):
        """
        Initialize the Kriging class with a variogram model.
        :param variogram_model: A callable variogram model function.
        """
        self.variogram_model = variogram_model
        self.coordinates = None
        self.values = None

    def fit(self, coordinates, values):
        """
        Fit the Kriging model to the given data.
        :param coordinates: Array of shape (n_samples, n_dimensions) with the coordinates of the data points.
        :param values: Array of shape (n_samples,) with the values at the data points.
        """
        self.coordinates = np.array(coordinates)
        self.values = np.array(values)

    def predict(self, points):
        """
        Predict values at the given points using the Kriging model.
        :param points: Array of shape (n_points, n_dimensions) with the coordinates of the points to predict.
        :return: Array of predicted values at the given points.
        """
        if self.coordinates is None or self.values is None:
            raise ValueError("The model must be fitted before prediction.")

        points = np.array(points)
        distances = cdist(points, self.coordinates)
        covariance_matrix = self.variogram_model(cdist(self.coordinates, self.coordinates))
        covariance_vector = self.variogram_model(distances)

        # Add a small nugget effect to the diagonal for numerical stability
        nugget = 1e-10
        covariance_matrix += np.eye(covariance_matrix.shape[0]) * nugget

        weights = np.dot(covariance_vector, inv(covariance_matrix))
        predictions = np.dot(weights, self.values)

        return predictions

# Example variogram model (spherical model)
def spherical_variogram(h, range_=1.0, sill=1.0, nugget=0.0):
    h = np.array(h)
    gamma = np.zeros_like(h)
    mask = h <= range_
    gamma[mask] = nugget + sill * (1.5 * (h[mask] / range_) - 0.5 * (h[mask] / range_)**3)
    gamma[~mask] = nugget + sill
    return sill - gamma

# Additional variogram models
# Monte Carlo variogram model
def monte_carlo_variogram(h, range_=1.0, sill=1.0, nugget=0.0, n_samples=1000):
    h = np.array(h)
    gamma = np.zeros_like(h)
    for i in range(len(h)):
        samples = np.random.uniform(0, range_, n_samples)
        gamma[i] = nugget + sill * np.mean(1.5 * (samples / range_) - 0.5 * (samples / range_)**3)
    gamma[h > range_] = nugget + sill
    return sill - gamma

# Latin Hypercube variogram model
def latin_hypercube_variogram(h, range_=1.0, sill=1.0, nugget=0.0, n_samples=1000):
    h = np.array(h)
    gamma = np.zeros_like(h)
    for i in range(len(h)):
        samples = np.linspace(0, range_, n_samples)
        np.random.shuffle(samples)
        gamma[i] = nugget + sill * np.mean(1.5 * (samples / range_) - 0.5 * (samples / range_)**3)
    gamma[h > range_] = nugget + sill
    return sill - gamma

# Exponential variogram model
def exponential_variogram(h, range_=1.0, sill=1.0, nugget=0.0):
    h = np.array(h)
    gamma = nugget + sill * (1 - np.exp(-3 * h / range_))
    return sill - gamma

# Gaussian variogram model
def gaussian_variogram(h, range_=1.0, sill=1.0, nugget=0.0):
    h = np.array(h)
    gamma = nugget + sill * (1 - np.exp(-3 * (h / range_)**2))
    return sill - gamma

# Linear variogram model
def linear_variogram(h, range_=1.0, sill=1.0, nugget=0.0):
    h = np.array(h)
    gamma = np.minimum(nugget + sill * (h / range_), sill + nugget)
    return sill - gamma


# Example usage
if __name__ == "__main__":
    # Define sample data
    coordinates = [(0, 0), (1, 0), (0, 1), (1, 1)]
    values = [1.0, 2.0, 1.5, 2.5]

    # Initialize and fit the Kriging model
    kriging = Kriging(variogram_model=lambda h: spherical_variogram(h, range_=1.5, sill=1.0, nugget=0.1))
    kriging.fit(coordinates, values)

    # Predict at new points
    new_points = [(0.5, 0.5), (1.5, 1.5)]
    predictions = kriging.predict(new_points)
    print("Predictions at new points:", predictions)