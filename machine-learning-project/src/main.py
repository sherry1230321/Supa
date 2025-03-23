import numpy as np
import pandas as pd
from src.data_preprocessing import preprocess_data
from src.model_training import train_models
from src.model_evaluation import evaluate_models
from src.real_time_analysis import perform_real_time_analysis
from src.deep_learning.rnn import RNN
from src.deep_learning.cnn import CNN
from src.deep_learning.lstm import LSTM
from src.clustering.kmeans import KMeans

def main():
    # Step 1: Load and preprocess the data
    raw_data_path = 'data/raw/data_file.csv'  # Replace with actual data file path
    processed_data = preprocess_data(raw_data_path)

    # Step 2: Train models
    trained_models = train_models(processed_data)

    # Step 3: Evaluate models
    evaluation_results = evaluate_models(trained_models, processed_data)

    # Step 4: Perform real-time analysis
    perform_real_time_analysis(trained_models)

if __name__ == "__main__":
    main()