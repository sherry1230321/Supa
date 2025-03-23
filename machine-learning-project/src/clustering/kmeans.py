import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class KMeansClustering:

    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.scaler = StandardScaler()

    def fit(self, X):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def get_cluster_centers(self):
        return self.model.cluster_centers_

    def get_labels(self):
        return self.model.labels_

    def inertia(self):
        return self.model.inertia_