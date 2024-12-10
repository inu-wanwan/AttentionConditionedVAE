import pandas as pd
import os
from utils import load_config

"""
make_sample_dataset.py
Creates a sample dataset from the canonicalized smiles dataset.
"""

path_config = load_config('filepath.yml')

smiles_file = os.path.join(path_config['data']['preprocessed'], 'canonicalized_smiles.csv')
output_file = os.path.join(path_config['data']['samples'], 'sample_dataset.csv')

df = pd.read_csv(smiles_file)

sample_df = df.sample(n=1000, random_state=42)

sample_df.to_csv(output_file, index=False)

print(f"Sample dataset saved to {output_file}")