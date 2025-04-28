import numpy as np
from sklearn.preprocessing import *
from dataclasses import dataclass
import joblib

@dataclass
class NormalizerParams:
    scaler_name: str = "MinMaxScaler"
    feature_range: tuple = (0, 1)  # Default for MinMaxScaler
    copy: bool = True  # Default for MinMaxScaler
    clip: bool = False  # Default for MinMaxScaler
    # Defaults for StandardScaler
    with_mean: bool = True  # Default for StandardScaler
    with_std: bool = True  # Default for StandardScaler

    # Defaults for RobustScaler
    quantile_range: tuple = (25.0, 75.0)  # Default for RobustScaler
    unit_variance: bool = False  # Default for RobustScaler

    # Defaults for MaxAbsScaler
    copy: bool = True  # Default for MaxAbsScaler

class DataNormalizer:
    def __init__(self, scaler_name="MinMaxScaler", **kwargs):
        """
        Initializes the DataNormalizer with a specified scaler from sklearn.
        
        :param scaler_name: Name of the scaler class from sklearn.preprocessing.
        :param kwargs: Additional keyword arguments for the scaler.
        """
        scaler_class = getattr(__import__('sklearn.preprocessing', fromlist=[scaler_name]), scaler_name)
        self.scaler = scaler_class(**kwargs)

    def fit(self, data):
        """
        Fits the scaler to the data.
        
        :param data: A 2D numpy array or list of lists to fit the scaler.
        """
        self.scaler.fit(data)

    def transform(self, data):
        """
        Transforms the data using the fitted scaler.
        
        :param data: A 2D numpy array or list of lists to transform.
        :return: Transformed data as a numpy array.
        """
        return self.scaler.transform(data)

    def fit_transform(self, data):
        """
        Fits the scaler to the data and transforms it.
        
        :param data: A 2D numpy array or list of lists to fit and transform.
        :return: Transformed data as a numpy array.
        """
        return self.scaler.fit_transform(data)
    
    def inverse_transform(self, data):
        """
        Inverse transforms the data using the fitted scaler.
        
        :param data: A 2D numpy array or list of lists to inverse transform.
        :return: Inverse transformed data as a numpy array.
        """
        return self.scaler.inverse_transform(data)
    
    def load(self, file_path):
        """
        Loads a scaler from a .pkl file if scaler_name has '.pkl' in it.
        
        :param file_path: Path to the .pkl file containing the saved scaler.
        """
        if ".pkl" in self.scaler.__class__.__name__:
            self.scaler = joblib.load(file_path)
        else:
            raise ValueError("Scaler name does not indicate a .pkl file.")
    
    def save(self, file_path):
        """
        Saves the fitted scaler to a .pkl file.
        
        :param file_path: Path where the scaler should be saved.
        """
        joblib.dump(self.scaler, file_path)
    


# Example usage
if __name__ == "__main__":
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    normalizer = DataNormalizer(scaler_name="StandardScaler")
    normalized_data = normalizer.fit_transform(data)
    print("Normalized Data:")
    print(normalized_data)
    