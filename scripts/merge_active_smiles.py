import os
import csv
from utils import load_config

"""
merge_active_smiles.py
Merges the active SMILES strings from the uniprot dataset.
{UNIPROT_ID}.ism files are merged into a single CSV file.
"""

path_config = load_config('filepath.yml')

dude_dir = path_config['data']['DUD-E']
output_dir = path_config['data']['smiles']

# load the list of proteins
protein_list_file = os.path.join(dude_dir, 'target_proteins.txt')
with open(protein_list_file, 'r') as f:
    protein_list = f.read().splitlines()

protein_paths = [os.path.join(dude_dir, protein) for protein in protein_list]

results = []
