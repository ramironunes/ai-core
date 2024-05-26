# -*- coding: utf-8 -*-
# @Author: Ramiro Luiz Nunes
# @Date:   2024-05-26 18:22:13
# @Last Modified by:   Ramiro Luiz Nunes
# @Last Modified time: 2024-05-26 18:26:23


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from io import BytesIO
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.drawing.image import Image


def perform_eda(attributes: pd.DataFrame, classes: pd.Series) -> None:
    """
    Perform EDA on the dataset.

    Args:
    - attributes: DataFrame containing the attributes.
    - classes: Series containing the classes.
    """
    print("First 5 rows of the dataset:")
    print(attributes.head())
    print("Shape of the dataset:", attributes.shape)

    print("Statistical information:")
    print(attributes.describe())

    print("Missing values in the dataset:")
    print(attributes.isnull().sum())

    print("Class distribution:")
    class_counts = classes.value_counts()
    print(class_counts)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    class_counts.plot(kind='bar', title='Class Distribution')

    plt.subplot(1, 3, 2)
    sns.boxplot(data=attributes)
    plt.title('Box Plot of Attributes')

    plt.subplot(1, 3, 3)
    sns.heatmap(attributes.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')

    plt.tight_layout()
    plt.show()

def save_eda_to_excel(attributes: pd.DataFrame, classes: pd.Series, filename: str) -> None:
    """
    Save EDA results to an Excel file.

    Args:
    - attributes: DataFrame containing the attributes.
    - classes: Series containing the classes.
    - filename: The name of the Excel file.
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "EDA"

    for r in dataframe_to_rows(attributes.head(), index=False, header=True):
        ws.append(r)
    ws.append([])

    stats = attributes.describe()
    for r in dataframe_to_rows(stats, index=True, header=True):
        ws.append(r)
    ws.append([])

    missing = attributes.isnull().sum().reset_index()
    missing.columns = ['Column', 'Missing Values']
    for r in dataframe_to_rows(missing, index=False, header=True):
        ws.append(r)
    ws.append([])

    class_counts = classes.value_counts().reset_index()
    class_counts.columns = ['Class', 'Count']
    for r in dataframe_to_rows(class_counts, index=False, header=True):
        ws.append(r)
    ws.append([])

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    class_counts.plot(kind='bar', x='Class', y='Count', title='Class Distribution', ax=plt.gca())

    plt.subplot(1, 3, 2)
    sns.boxplot(data=attributes, ax=plt.gca())
    plt.title('Box Plot of Attributes')

    plt.subplot(1, 3, 3)
    sns.heatmap(attributes.corr(), annot=True, cmap='coolwarm', ax=plt.gca())
    plt.title('Correlation Matrix')

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image(buf)
    ws.add_image(img, 'A15')

    wb.save(filename)
