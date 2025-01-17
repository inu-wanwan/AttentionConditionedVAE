import pandas as pd
import os
from utils import load_config
from sklearn.model_selection import train_test_split

"""
make_sample_dataset.py
Creates a sample dataset for local model from the canonicalized smiles dataset.
"""

path_config = load_config('filepath.yml')

smiles_file = os.path.join(path_config['data']['preprocessed'], 'dataset_final.csv')
test_dir = path_config['data']['test']
train_dir = path_config['data']['train']
val_dir = path_config['data']['val']

smiles_df = pd.read_csv(smiles_file)

diverse_proteins = ['AKT1', 'AMPC', 'CP3A4', 'CXCR4', 'GCR', 'HIVPR', 'HIVRT', 'KIF11']

for protein in diverse_proteins:
    test_file = os.path.join(test_dir, f'test_{protein}.csv')
    train_file = os.path.join(train_dir, f'train_{protein}.csv')
    val_file = os.path.join(val_dir, f'val_{protein}.csv')

    protein_df = smiles_df[smiles_df['Target'] == protein]
    train_df, test_val_df = train_test_split(protein_df, test_size=0.2, random_state=42)
    test_df, val_df = train_test_split(test_val_df, test_size=0.5, random_state=42)

    print(f"Train dataset size for {protein}: {len(train_df)}")
    print(f"Test dataset size for {protein}: {len(test_df)}")
    print(f"Validation dataset size for {protein}: {len(val_df)}")

    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    val_df.to_csv(val_file, index=False)

    print(f"Train dataset saved to {train_file}")
    print(f"Test dataset saved to {test_file}")
    print(f"Validation dataset saved to {val_file}")