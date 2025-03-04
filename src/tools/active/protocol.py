import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class ActiveLearning:
    def __init__(self, model, X_pool, y_pool, X_test, y_test, initial_size=10, query_size=5):
        self.model = model
        self.X_pool = X_pool
        self.y_pool = y_pool
        self.X_test = X_test
        self.y_test = y_test
        self.initial_size = initial_size
        self.query_size = query_size
        self.X_train, self.y_train = self._initialize_training_set()

    def _initialize_training_set(self):
        indices = np.random.choice(range(len(self.X_pool)), size=self.initial_size, replace=False)
        X_train = self.X_pool[indices]
        y_train = self.y_pool[indices]
        self.X_pool = np.delete(self.X_pool, indices, axis=0)
        self.y_pool = np.delete(self.y_pool, indices, axis=0)
        return X_train, y_train

    def query(self):
        self.model.fit(self.X_train, self.y_train)
        pool_predictions = self.model.predict_proba(self.X_pool)
        uncertainty = np.max(pool_predictions, axis=1)
        query_indices = np.argsort(uncertainty)[:self.query_size]
        return query_indices

    def update_training_set(self, query_indices):
        self.X_train = np.vstack((self.X_train, self.X_pool[query_indices]))
        self.y_train = np.hstack((self.y_train, self.y_pool[query_indices]))
        self.X_pool = np.delete(self.X_pool, query_indices, axis=0)
        self.y_pool = np.delete(self.y_pool, query_indices, axis=0)

    def evaluate(self):
        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)
        return accuracy_score(self.y_test, predictions)

    def run(self, iterations=10):
        for i in range(iterations):
            query_indices = self.query()
            self.update_training_set(query_indices)
            accuracy = self.evaluate()
            print(f"Iteration {i+1}/{iterations}, Accuracy: {accuracy:.4f}")

# Example usage:
# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier()
# X_pool, y_pool, X_test, y_test = ... # Load your data here
# active_learner = ActiveLearning(model, X_pool, y_pool, X_test, y_test)
# active_learner.run()