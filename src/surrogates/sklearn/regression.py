from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import SGDRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.base import ClassifierMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB


#To-do: Add Cross Decmposition, Kernel Ridge, ElasticNet, PassiveAggressiveRegressor, Perceptron, TheilSenRegressor, HuberRegressor, RANSACRegressor, OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV, BayesianRidge, ARDRegression, PoissonRegressor, TweedieRegressor
# Add in Bagging and Voting Regressor, Bagging and Voting Classifier
# PLSCanoncial, PLSSVD, PLS Regression
# Define multiple linear regression model
class MultipleLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

class LogisticRegressionModel(BaseEstimator, RegressorMixin):
    def __init__(self, penalty='l2', C=1.0, max_iter=100):
        self.model = LogisticRegression(penalty=penalty, C=C, max_iter=max_iter)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
# Define ridge regression model
class RidgeRegression(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

# Define lasso regression model
class LassoRegression(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0):
        self.model = Lasso(alpha=alpha)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


# Define SGD regression model
class SGDRegression(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=0.0001, max_iter=1000, tol=1e-3):
        self.model = SGDRegressor(alpha=alpha, max_iter=max_iter, tol=tol)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

# Define Naive Bayes classification model
class NaiveBayesClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model = GaussianNB()

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
# Define Multinomial Naive Bayes classification model
class MultinomialNaiveBayesClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        self.model = MultinomialNB(alpha=alpha, fit_prior=fit_prior, class_prior=class_prior)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

# Define Bernoulli Naive Bayes classification model
class BernoulliNaiveBayesClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None):
        self.model = BernoulliNB(alpha=alpha, binarize=binarize, fit_prior=fit_prior, class_prior=class_prior)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

# Define Complement Naive Bayes classification model
class ComplementNaiveBayesClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None, norm=False):
        self.model = ComplementNB(alpha=alpha, fit_prior=fit_prior, class_prior=class_prior, norm=norm)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

# Define Categorical Naive Bayes classification model
class CategoricalNaiveBayesClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        self.model = CategoricalNB(alpha=alpha, fit_prior=fit_prior, class_prior=class_prior)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
# Define isotonic regression model
class IsotonicRegressionModel(BaseEstimator, RegressorMixin):
    def __init__(self, y_min=None, y_max=None, increasing=True):
        self.model = IsotonicRegression(y_min=y_min, y_max=y_max, increasing=increasing)

    def fit(self, X, y):
        self.model.fit(X.ravel(), y)
        return self

    def predict(self, X):
        return self.model.predict(X.ravel())