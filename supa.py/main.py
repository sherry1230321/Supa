import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import os
import tempfile

class MachineLearningAI:

    def __init__(self, classifier=None, param_grid=None):
        self.classifier = classifier if classifier is not None else MLPClassifier()
        self.param_grid = param_grid
        self.pipeline = None
        self.grid_search = None
        self.best_params = None
        self.best_score = None

    def generate_dataset(self, n_samples=1000, n_features=20, test_size=0.25, random_state=42):
        X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=2, n_classes=2, random_state=random_state)
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def configure_pipeline(self, steps):
        self.pipeline = Pipeline(steps)

    def generate_and_train_classifier(self, X_train, y_train, cv=5):
        if self.pipeline is None:
            self.configure_pipeline([('scaler', StandardScaler()), ('poly', PolynomialFeatures()), ('classifier', self.classifier)])
        if self.param_grid is not None:
            self.grid_search = GridSearchCV(self.pipeline, self.param_grid, cv=cv, n_jobs=-1)
            self.grid_search.fit(X_train, y_train)
            print(f'Best parameters found: {self.grid_search.best_params_}')
            self.best_params = self.grid_search.best_params_
            self.pipeline = self.grid_search.best_estimator_
        else:
            scores = cross_val_score(self.pipeline, X_train, y_train, cv=cv, n_jobs=-1)
            self.pipeline.fit(X_train, y_train)
            return np.mean(scores)

    def evaluate_classifier(self, X_test, y_test):
        y_pred = self.pipeline.predict(X_test)
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        return accuracy_score(y_test, y_pred)

    def save_model(self, filename):
        with tempfile.TemporaryDirectory() as tmpdirname:
            temp_path = os.path.join(tmpdirname, filename)
            joblib.dump(self.pipeline, temp_path)
            os.rename(temp_path, filename)
        print(f'Model saved to {filename}')

    def load_model(self, filename):
        with tempfile.TemporaryDirectory() as tmpdirname:
            temp_path = os.path.join(tmpdirname, filename)
            self.pipeline = joblib.load(temp_path)
        print(f'Model loaded from {filename}')

# Set up classifiers and parameter grids to use
classifier_params = {
    'MLP': (MLPClassifier(), {
        'classifier__hidden_layer_sizes': [(50,), (100,)],
        'classifier__activation': ['tanh', 'relu', 'logistic'],
        'classifier__solver': ['sgd', 'adam', 'lbfgs'],
        'classifier__alpha': [0.0001, 0.05],
        'classifier__learning_rate': ['constant', 'adaptive', 'invscaling'],
        'classifier__learning_rate_init': [0.001, 0.01, 0.1, 0.5, 1],
        'classifier__max_iter': [100, 1000],
        'classifier__momentum': [0.9, 0.99],
        'classifier__nesterovs_momentum': [True, False],
        'classifier__early_stopping': [True, False],
        'classifier__validation_fraction': [0.1, 0.2],
        'classifier__beta_1': [0.9, 0.99],
        'classifier__beta_2': [0.9, 0.99],
        'classifier__epsilon': [1e-08, 1e-07],
        'classifier__n_iter_no_change': [10, 20, 30],
        'classifier__shuffle': [True, False],
        'classifier__random_state': [0, 42, 100],
    }),
    'Decision Tree': (DecisionTreeClassifier(), {
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__min_weight_fraction_leaf': [0.0, 0.1, 0.2],
        'classifier__criterion': ['gini', 'entropy'],
        'classifier__splitter': ['best', 'random'],
        'classifier__max_features': ['auto', 'sqrt', 'log2'],
        'classifier__max_leaf_nodes': [None, 5, 10, 20],
        'classifier__min_impurity_decrease': [0.0, 0.1, 0.2],
        'classifier__presort': [True, False],
        'classifier__random_state': [0, 42, 100],
    }),
    'Random Forest': (RandomForestClassifier(), {
        'classifier__n_estimators': [50, 100, 200, 500],
        'classifier__max_features': ['auto', 'sqrt', 'log2'],
        'classifier__max_depth': [None, 5, 10, 15, 20],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4, 8],
        'classifier__bootstrap': [True, False],
        'classifier__oob_score': [True, False],
        'classifier__max_leaf_nodes': [None, 5, 10, 20],
        'classifier__min_impurity_decrease': [0.0, 0.1, 0.2],
        'classifier__warm_start': [True, False],
        'classifier__class_weight': ['balanced', None],
        'classifier__random_state': [0, 42, 100],
    }),
    'SVM': (SVC(), {
        'classifier__C': [0.1, 1, 10],
        'classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'classifier__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'classifier__degree': [2, 3, 4],
        'classifier__coef0': [0.0, 0.1, 0.5],
        'classifier__shrinking': [True, False],
        'classifier__probability': [True, False],
        'classifier__tol': [1e-3, 1e-4, 1e-5],
        'classifier__cache_size': [200, 400, 800],
        'classifier__max_iter': [-1],
        'classifier__decision_function_shape': ['ovo', 'ovr'],
        'classifier__break_ties': [True, False],
        'classifier__random_state': [0, 42, 100],
    }),
    'Logistic Regression': (LogisticRegression(), {
        'classifier__C': [0.1, 1, 10, 100, 1000, 10000],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__dual': [True, False],
        'classifier__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'classifier__max_iter': [100, 1000, 10000],
        'classifier__fit_intercept': [True, False],
        'classifier__intercept_scaling': [True, False],
        'classifier__warm_start': [True, False],
        'classifier__n_jobs': [-1],
        'classifier__l1_ratio': [0.1, 0.5, 0.9],
        'classifier__class_weight': ['balanced', None],
        'classifier__random_state': [0, 42, 100],
        'classifier__tol': [1e-4, 1e-3, 1e-2],
        'classifier__verbose': [0, 1, 2],
        'classifier__multi_class': ['auto', 'ovr', 'multinomial'],
    }),
}

def save_best_models(classifier_params):
    for name, (clf, params) in classifier_params.items():
        print(f'\nTraining and evaluating {name}')
        ai = MachineLearningAI(classifier=clf, param_grid=params)
        X_train, X_test, y_train, y_test = ai.generate_dataset()

        ai.generate_and_train_classifier(X_train, y_train)
        accuracy = ai.evaluate_classifier(X_test, y_test)
        print(f'{name} Model Accuracy: {accuracy}')

        # Save the best model
        model_file = f'best_{name.lower().replace(" ", "_")}_model.joblib'
        ai.save_model(model_file)

# Call the function to save the best models
save_best_models(classifier_params)

class DeepLearningAI(MachineLearningAI):

    def __init__(self, classifier=None, param_grid=None):
        super().__init__(classifier, param_grid)

    def configure_deep_learning_model(self, input_dim):
        self.classifier = Sequential([
            Dense(128, input_dim=input_dim, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid'),
        ])
        self.classifier.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    def train_deep_learning_model(self, X_train, y_train, epochs=20, batch_size=32):
        print("Starting training...")
        self.classifier.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        print("Training completed.")

    def evaluate_deep_learning_model(self, X_test, y_test):
        evaluation = self.classifier.evaluate(X_test, y_test)
        return evaluation

# Integration into the existing workflow
def train_and_save_deep_learning_model():
    ai = DeepLearningAI()
    X_train, X_test, y_train, y_test = ai.generate_dataset()

    ai.configure_deep_learning_model(input_dim=20)
    ai.train_deep_learning_model(X_train, y_train, epochs=20, batch_size=10)
    evaluation = ai.evaluate_deep_learning_model(X_test, y_test)
    print(f'Deep Learning Model Accuracy: {evaluation[1]}')

    # Save the deep learning model
    model_file = 'deep_learning_model.h5'
    ai.classifier.save(model_file)
    print(f'Model saved to {model_file}')

# Call the function to train and save the deep learning model
train_and_save_deep_learning_model()
