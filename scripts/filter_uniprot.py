from utils import load_config
import os
import pandas as pd

"""
Filter the uniprot data to check which uniprot data is available in the alphafold directory
"""

config = load_config("filepath.yml")
alphafold_dir = config['data']['alphafold']
dude_dir = config['data']['DUD-E']
smiles_dir = config['data']['smiles']

# Load the uniprot data
df = pd.read_csv(os.path.join(dude_dir, 'targets_pdb_ids.csv'))
uniprot = df['UniProt ID']

# Filter the uniprot data
exsisting_uniprot = df[uniprot.apply(lambda uniprot: os.path.exists(os.path.join(alphafold_dir, uniprot)))]
missing_uniprot = df[~uniprot.apply(lambda uniprot: os.path.exists(os.path.join(alphafold_dir, uniprot)))]

exsisting_uniprot.to_csv(os.path.join(smiles_dir, 'available_uniprot.csv'), index=False)
missing_uniprot.to_csv(os.path.join(smiles_dir, 'missing_uniprot.csv'), index=False)

