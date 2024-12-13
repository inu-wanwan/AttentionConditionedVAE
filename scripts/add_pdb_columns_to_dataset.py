import os
import pandas as pd
from utils import load_config

"""
add_pdb_columns_to_dataset.py
Adds PDB columns to the dataset.
"""

path_config = load_config('filepath.yml')

dataset_file = os.path.join(path_config['data']['preprocessed'], 'filtered_by_available_uniprot.csv')
pdb_file = os.path.join(path_config['data']['protein'], 'available_uniprot.csv')
output_file = os.path.join(path_config['data']['preprocessed'], 'dataset_final.csv')

dataset = pd.read_csv(dataset_file)
pdb_mapping = pd.read_csv(pdb_file)

dataset['Protein'] = dataset['Protein'].str.upper()
pdb_mapping['Target'] = pdb_mapping['Target'].str.upper()

dataset = dataset.merge(pdb_mapping, left_on='Protein', right_on='Target', how='left')
dataset = dataset.drop(columns=['Protein'])

dataset.to_csv(dataset_file, index=False)

print(f"Dataset with PDB columns saved to {dataset_file}")