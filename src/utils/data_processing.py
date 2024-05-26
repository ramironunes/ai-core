# -*- coding: utf-8 -*-
# @Author: Ramiro Luiz Nunes
# @Date:   2024-05-26 18:21:30
# @Last Modified by:   Ramiro Luiz Nunes
# @Last Modified time: 2024-05-26 18:27:35


import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_file_size(file_path: str) -> int:
    """
    Get the size of the file in bytes.

    Args:
    - file_path: The path to the file.

    Returns:
    - The size of the file in bytes.
    """
    return os.path.getsize(file_path)

def read_csv_in_chunks(file_path: str, chunksize: int = 10000) -> pd.DataFrame:
    """
    Read a CSV file in chunks and concatenate into a single DataFrame.

    Args:
    - file_path: The path to the CSV file.
    - chunksize: The number of rows per chunk.

    Returns:
    - A DataFrame containing all the data from the CSV file.
    """
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)

def read_csv_no_header(file_path: str) -> tuple:
    data = pd.read_csv(file_path, header=None)
    attributes = data.iloc[:, :-1]
    classes = data.iloc[:, -1]
    return attributes, classes

def preprocess_data(attributes: pd.DataFrame, classes: pd.Series) -> tuple:
    X_train, X_test, y_train, y_test = train_test_split(attributes, classes, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test
