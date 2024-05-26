# -*- coding: utf-8 -*-
# @Author: Ramiro Luiz Nunes
# @Date:   2024-05-26 18:28:00
# @Last Modified by:   Ramiro Luiz Nunes
# @Last Modified time: 2024-05-26 18:28:35


import os
import pandas as pd


def analyze_datasets(directory_path: str) -> pd.DataFrame:
    """
    Analyze all datasets in the given directory and return a summary DataFrame.

    Args:
    - directory_path: The path to the directory containing the CSV files.

    Returns:
    - A DataFrame containing the analysis summary of all datasets.
    """
    summary = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            file_size = os.path.getsize(file_path)
            
            data = pd.read_csv(file_path, nrows=1000)  # Read first 1000 rows for quick analysis
            num_rows, num_cols = data.shape
            class_distribution = data.iloc[:, -1].value_counts().to_dict()
            
            summary.append({
                'filename': filename,
                'file_size': file_size,
                'num_rows': num_rows,
                'num_cols': num_cols,
                'class_distribution': class_distribution
            })

    return pd.DataFrame(summary)
