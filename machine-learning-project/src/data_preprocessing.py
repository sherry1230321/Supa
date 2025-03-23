import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    data = data.dropna()  # Remove missing values
    # Additional cleaning steps can be added here
    return data

def preprocess_data(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y

def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def split_data(X, y, test_size=0.25, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test