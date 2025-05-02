from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import SGDRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.base import ClassifierMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB
from sklearn.cross_decomposition import PLSCanonical, PLSRegression, PLSSVD
from sklearn.ensemble import BaggingRegressor, BaggingClassifier, VotingRegressor, VotingClassifier


#To-do: Add Cross Decmposition, Kernel Ridge, ElasticNet, PassiveAggressiveRegressor, TheilSenRegressor, HuberRegressor, RANSACRegressor, OrthogonalMatchingPursuit
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
    
# Define PLSCanonical model
class PLSCanonicalModel(BaseEstimator, RegressorMixin):
    def __init__(self, n_components=2, scale=True, max_iter=500, tol=1e-06):
        self.model = PLSCanonical(n_components=n_components, scale=scale, max_iter=max_iter, tol=tol)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

# Define PLSRegression model
class PLSRegressionModel(BaseEstimator, RegressorMixin):
    def __init__(self, n_components=2, scale=True, max_iter=500, tol=1e-06):
        self.model = PLSRegression(n_components=n_components, scale=scale, max_iter=max_iter, tol=tol)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

# Define PLSSVD model
class PLSSVDModel(BaseEstimator, RegressorMixin):
    def __init__(self, n_components=2):
        self.model = PLSSVD(n_components=n_components)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
    
# Define Bagging Regressor model
class BaggingRegressorModel(BaseEstimator, RegressorMixin):
    def __init__(self, base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=None, random_state=None, verbose=0):
        self.model = BaggingRegressor(base_estimator=base_estimator, n_estimators=n_estimators, max_samples=max_samples, max_features=max_features, bootstrap=bootstrap, bootstrap_features=bootstrap_features, n_jobs=n_jobs, random_state=random_state, verbose=verbose)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

# Define Bagging Classifier model
class BaggingClassifierModel(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=None, random_state=None, verbose=0):
        self.model = BaggingClassifier(base_estimator=base_estimator, n_estimators=n_estimators, max_samples=max_samples, max_features=max_features, bootstrap=bootstrap, bootstrap_features=bootstrap_features, n_jobs=n_jobs, random_state=random_state, verbose=verbose)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

# Define Voting Regressor model
class VotingRegressorModel(BaseEstimator, RegressorMixin):
    def __init__(self, estimators, weights=None, n_jobs=None, verbose=False):
        self.model = VotingRegressor(estimators=estimators, weights=weights, n_jobs=n_jobs, verbose=verbose)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

# Define Voting Classifier model
class VotingClassifierModel(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators, voting='hard', weights=None, n_jobs=None, flatten_transform=True, verbose=False):
        self.model = VotingClassifier(estimators=estimators, voting=voting, weights=weights, n_jobs=n_jobs, flatten_transform=flatten_transform, verbose=verbose)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)