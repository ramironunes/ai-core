# -*- coding: utf-8 -*-
# @Author: Ramiro Luiz Nunes
# @Date:   2024-05-26 18:22:39
# @Last Modified by:   Ramiro Luiz Nunes
# @Last Modified time: 2024-05-26 18:22:44


import pandas as pd

from openpyxl import Workbook


def save_results_to_excel(results: pd.DataFrame, output_path: str) -> None:
    """
    Save the model accuracies and comparisons to an Excel file.

    Args:
    - results: DataFrame containing the results.
    - output_path: The path where the Excel file will be saved.
    """
    results.to_excel(output_path, index=False)

    # Create a summary sheet for comparisons
    with pd.ExcelWriter(output_path, engine='openpyxl', mode='a') as writer:
        summary_sheet = pd.DataFrame()
        for model in results.columns[:-1]:  # Exclude the 'filename' column
            summary = results.groupby('filename')[model].mean().reset_index()
            summary.columns = ['Dataset', f'{model} Accuracy']
            summary_sheet = pd.concat([summary_sheet, summary], axis=1)

        summary_sheet.to_excel(writer, sheet_name='Summary', index=False)
