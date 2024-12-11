from utils import load_config
import os
import pandas as pd

config = load_config("filepath.yml")
smiles_dir = config['data']['smiles']
preprocessed_dir = config['data']['preprocessed']

# Load available uniprot data
available_target = pd.read_csv(os.path.join(smiles_dir, 'exsisting_uniprot.csv'))
available_target = available_target['Target'].apply(lambda x: x.lower())

# Load dataset and filter by available uniprot data
df = pd.read_csv(os.path.join(preprocessed_dir, 'ligand_with_docking_score.csv'))
df = df[df['Protein'].isin(available_target)]
df.to_csv(os.path.join(preprocessed_dir, 'dataset_final.csv'), index=False)

