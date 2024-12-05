import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from utils import load_config
from tqdm import tqdm
    
"""
canonicalize_smiles.py
Canonicalizes the SMILES strings in the dataset.
"""

path_config = load_config('filepath.yml')
smiles_path = path_config['data']['smiles']
output_path = path_config['data']['preprocessed']

def canonicalize_smiles(smiles):
    """
    Canonicalizes a SMILES string.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception as e:
        print(f"Error processing SMILES string {smiles}: {e}")

# load the SMILES strings
df = pd.read_csv(os.path.join(smiles_path, 'smiles.csv'))

tqdm.pandas(desc="Canonicalizing SMILES")
df["Canonical_SMILES"] = df["SMILES"].progress_apply(canonicalize_smiles)  
df = df.dropna(subset=["Canonical_SMILES"])

# save the canonicalized SMILES strings
df.to_csv(os.path.join(output_path, 'canonical_smiles.csv'), index=False)