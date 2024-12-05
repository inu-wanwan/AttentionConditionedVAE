import os
import csv
import yaml
from utils import load_config

"""
extract_smiles.py
Extracts the SMILES strings from the DUD-E dataset and writes them to a CSV file.
"""

path_config = load_config('filepath.yml')

dude_dir = path_config['data']['DUD-E']
output_dir = path_config['data']['smiles']

# load the list of proteins
protein_list_file = os.path.join(dude_dir, 'target_proteins.txt')
with open(protein_list_file, 'r') as f:
    protein_list = f.read().splitlines()

protein_paths = [os.path.join(dude_dir, protein) for protein in protein_list]
protein_num = len(protein_list)

results = []

# extract the SMILES strings from the .ism files
for protein_path in protein_paths:
    protein_name = os.path.basename(protein_path)
    
    for file_name in ["actives_final.ism", "decoys_final.ism"]:
        file_path = os.path.join(protein_path, file_name)
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    smiles = parts[0]
                    ligand_id = parts[-1]
                    results.append({
                        "SMILES": smiles,
                        "Ligand_id": ligand_id,
                        "Protein": protein_name,
                        "Active": file_name == "actives_final.ism"
                    })

# write the results to a CSV file
output_file = os.path.join(output_dir, 'smiles.csv')
with open(output_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"Extracted {len(results)} SMILES strings to {output_file}")