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

base_url = "https://dude.docking.org//targets/{target}/uniprot.txt"
# download the ligands for each uniprot ids
for protein in protein_list:
    uniprot_file = os.path.join(dude_dir, protein, 'uniprot.txt')

    # if no available uniprot.txt file, download it
    if not os.path.exists(uniprot_file):
        print(f"Downloading UniProt data for {protein} to {uniprot_file}")
        uniprot_url = base_url.format(target=protein)
        output_path = os.path.join(dude_dir, protein, 'uniprot.txt')
        if not requests.head(uniprot_url).ok:
            print(f"Skipping {protein}: no UniProt data available")
            continue
        wget.download(uniprot_url, out=output_path)

    with open(uniprot_file, 'r') as f:
        uniprot_ids = f.read().splitlines()
    
    # download the ligands for each uniprot id
    for uniprot_id in uniprot_ids:
        ligand_url = f"https://dude.docking.org//targets/{protein}/{uniprot_id}/actives_filtered.ism"
        output_path = os.path.join(dude_dir, protein, f'{uniprot_id}.ism')
        response = requests.get(ligand_url)

        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            with open(output_path, 'wb') as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            print(f"Downloaded {uniprot_id}.ism to {output_path}")
        else:
            print(f"Skipping {uniprot_id}: no ligand data available\n")
            print(f"status code: {response.status_code}\n")
        
