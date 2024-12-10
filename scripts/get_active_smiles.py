import wget
import os
import requests
from utils import load_config
from tqdm import tqdm

"""
get_active_smiles.py
Downloads the active ligands from the DUD-E dataset.
output_path = os.path.join(dude_dir, protein, f'{uniprot_id}.ism')
"""

path_config = load_config('filepath.yml')

dude_dir = path_config['data']['DUD-E']

# load the list of proteins
protein_list_file = os.path.join(dude_dir, 'target_proteins.txt')
with open(protein_list_file, 'r') as f:
    protein_list = f.read().splitlines()

# download the ligands for each uniprot ids
for protein in tqdm(protein_list, desc="Downloading active ligands"):
    protein_dir = os.path.join(dude_dir, protein)
    # get actives_final.ism
    actives_final_url = f"https://dude.docking.org//targets/{protein}/actives_final.ism"
    response = requests.get(actives_final_url)
    if response.status_code == 200:
        with open(os.path.join(protein_dir, 'actives_final.ism'), 'wb') as f:
            f.write(response.content)
    else:
        print(f"Error: {response.status_code} - {response.text}")
