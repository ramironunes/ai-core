# -*- coding: utf-8 -*-
# @Author: Ramiro Luiz Nunes
# @Date:   2024-05-21 12:38:10
# @Last Modified by:   Ramiro Luiz Nunes
# @Last Modified time: 2024-05-21 12:38:47


import pandas as pd


def read_csv_no_header(file_path):
    """
    Read a CSV file without header using pandas.
    
    Args:
    - file_path: The path to the CSV file.
    
    Returns:
    - data: Tuple containing attributes DataFrame and classes Series.
    """
    # Leitura do arquivo CSV sem cabeçalho
    data = pd.read_csv(file_path, header=None)
    
    # Separação dos atributos e classes
    attributes = data.iloc[:, :-1]  # Todas as colunas exceto a última
    classes = data.iloc[:, -1]       # Última coluna
    
    return attributes, classes

# Exemplo de uso:
file_path = 'misterioso.csv'
attributes, classes = read_csv_no_header(file_path)

print("Atributos:")
print(attributes.head())
print("\nClasses:")
print(classes.head())