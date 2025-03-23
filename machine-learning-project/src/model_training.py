import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class ModelTrainer:

    def __init__(self, model_type='random_forest', param_grid=None):
        self.model_type = model_type
        self.param_grid = param_grid
        self.model = self._initialize_model()
        self.pipeline = None

    def _initialize_model(self):
        if self.model_type == 'random_forest':
            return RandomForestClassifier()
        elif self.model_type == 'svm':
            return SVC()
        elif self.model_type == 'logistic_regression':
            return LogisticRegression()
        elif self.model_type == 'deep_learning':
            return Sequential([
                Dense(128, input_dim=20, activation='relu'),
                Dense(64, activation='relu'),
                Dense(1, activation='sigmoid'),
            ])
        else:
            raise ValueError("Unsupported model type")

    def configure_pipeline(self):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', self.model)
        ])

    def train(self, X_train, y_train):
        if self.pipeline is None:
            self.configure_pipeline()
        if self.param_grid:
            grid_search = GridSearchCV(self.pipeline, self.param_grid, cv=5, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            print(f'Best parameters found: {grid_search.best_params_}')
        else:
            self.pipeline.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Model Accuracy: {accuracy}')
        return accuracy

    def save_model(self, filename):
        joblib.dump(self.pipeline, filename)
        print(f'Model saved to {filename}')

    def load_model(self, filename):
        self.pipeline = joblib.load(filename)
        print(f'Model loaded from {filename}')