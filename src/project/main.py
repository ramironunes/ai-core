# -*- coding: utf-8 -*-
# @Author: Ramiro Luiz Nunes
# @Date:   2024-05-21 12:38:10
# @Last Modified by:   Ramiro Luiz Nunes
# @Last Modified time: 2024-05-26 18:17:29


import sys
import os
import pandas as pd
import logging
import gc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the src directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.knn_model import KNNModel
from src.models.svm_model import SVMModel
from src.models.mlp_model import MLPModel

from src.utils.data_processing import read_csv_in_chunks, preprocess_data, get_file_size
from src.utils.eda import perform_eda, save_eda_to_excel
from src.utils.file_handling import save_results_to_excel
from src.utils.data_analysis import analyze_datasets


# Define the reports directory
reports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'resources', 'reports'))

# Create the reports directory if it doesn't exist
os.makedirs(reports_dir, exist_ok=True)

def train_and_evaluate_selected_models(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> dict:
    """
    Train and evaluate selected models and return their accuracies.

    Args:
    - X_train: DataFrame containing the scaled attributes for training.
    - X_test: DataFrame containing the scaled attributes for testing.
    - y_train: Series containing the classes for training.
    - y_test: Series containing the classes for testing.

    Returns:
    - A dictionary containing the accuracy of each model.
    """
    models = [
        KNNModel(),
        SVMModel(),
        MLPModel()
    ]

    accuracies = {}
    for model in models:
        model_name = model.__class__.__name__
        logging.info(f"Training model: {model_name}")
        model.define_model()
        model.train(X_train, y_train)
        accuracy = model.evaluate(X_test, y_test)
        accuracies[model_name] = accuracy
        logging.info(f"Accuracy for {model_name}: {accuracy}")

    return accuracies

def process_all_datasets(directory_path: str, size_limit: int = 100 * 1024 * 1024) -> pd.DataFrame:
    """
    Process all CSV files in the given directory, train and evaluate models,
    and print the accuracy for each dataset. Also, save the results to a DataFrame.

    Args:
    - directory_path: The path to the directory containing the CSV files.
    - size_limit: The maximum file size to process (in bytes).

    Returns:
    - A DataFrame containing the results for all datasets.
    """
    results = []
    ignored_files = []

    if not os.path.exists(directory_path):
        logging.error(f"Directory does not exist: {directory_path}")
        return pd.DataFrame(results)

    analysis_summary = analyze_datasets(directory_path)
    analysis_summary_path = os.path.join(reports_dir, 'dataset_analysis.xlsx')
    analysis_summary.to_excel(analysis_summary_path, index=False)
    logging.info(f"Dataset analysis saved to {analysis_summary_path}")

    # Select three datasets for further analysis
    selected_datasets = analysis_summary.sort_values(by='file_size').head(3)['filename']

    for filename in selected_datasets:
        try:
            file_path = os.path.join(directory_path, filename)
            file_size = get_file_size(file_path)

            logging.info("*" * 50)
            logging.info(f"Processing file: {file_path} (size {file_size} bytes)")

            if file_size > size_limit:
                logging.warning(f"Skipping file {file_path} (size {file_size} bytes) - exceeds limit of {size_limit} bytes")
                ignored_files.append({'filename': filename, 'reason': 'File size exceeds limit'})
                continue

            # Read CSV in chunks using read_csv_in_chunks
            data = read_csv_in_chunks(file_path)
            attributes = data.iloc[:, :-1]
            classes = data.iloc[:, -1]

            # Execute EDA and save results to Excel
            eda_filename = os.path.join(reports_dir, os.path.basename(file_path).replace(".csv", "_EDA.xlsx"))
            save_eda_to_excel(attributes, classes, eda_filename)

            X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(attributes, classes)
            accuracies = train_and_evaluate_selected_models(X_train_scaled, X_test_scaled, y_train, y_test)
            accuracies['filename'] = filename
            results.append(accuracies)
            logging.info(f"Accuracies for {filename}: {accuracies}")

            # Save intermediate results to avoid loss
            intermediate_results_path = os.path.join(reports_dir, 'intermediate_accuracies.xlsx')
            pd.DataFrame(results).to_excel(intermediate_results_path, index=False)

            # Clear memory
            del data
            gc.collect()

        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}", exc_info=True)
            ignored_files.append({'filename': filename, 'reason': str(e)})

    # Save ignored files information
    ignored_files_path = os.path.join(reports_dir, 'ignored_files.xlsx')
    pd.DataFrame(ignored_files).to_excel(ignored_files_path, index=False)
    
    return pd.DataFrame(results)

# Directory containing the CSV files
directory_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'resources', 'dataset'))
logging.info(f"Directory path: {directory_path}")

try:
    # Process all datasets and save the results
    results_df = process_all_datasets(directory_path)
    output_path = os.path.join(reports_dir, 'model_accuracies.xlsx')
    save_results_to_excel(results_df, output_path)
    logging.info(f"Results saved to {output_path}")
except Exception as e:
    logging.error(f"Error in processing datasets: {e}", exc_info=True)
