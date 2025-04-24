from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor


# Define Gradient Boosted Trees model
class GradientBoostedTreesModel:
    def __init__(self, **kwargs):
        self.model = XGBRegressor(**kwargs)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

# Define Random Forest model
class RandomForestModel:
    def __init__(self, **kwargs):
        self.model = RandomForestRegressor(**kwargs)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)