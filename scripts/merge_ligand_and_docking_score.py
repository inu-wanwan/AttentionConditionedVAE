import os
import csv
import pandas as pd
from utils import load_config
from tqdm import tqdm

"""
merge_ligand_and_docking_score.py
Merges the ligand and docking scores into a single CSV file.
output_file : data/preprocessed/ligand_with_docking_score.csv
"""

path_config = load_config('filepath.yml')
dude_dir = path_config['data']['DUD-E']
output_dir = path_config['data']['preprocessed']

# load the list of proteins
protein_list_file = os.path.join(dude_dir, 'target_proteins.txt')
ligand_file = os.path.join(output_dir, 'canonicalized_smiles.csv')
docking_score_file = os.path.join(dude_dir, 'docking_scores.csv')

ligand_df = pd.read_csv(ligand_file)

with open(protein_list_file, 'r') as f:
    protein_list = f.read().splitlines()

if not os.path.exists(docking_score_file):
    docking_scores = []

    for protein in tqdm(protein_list, desc="Extracting docking scores"):
        interaction_file = os.path.join(dude_dir, protein, 'glide-dock_merged_best_pv.interaction')
        if not os.path.exists(interaction_file):
            print(f"File not found: {interaction_file}")
            continue

        with open(interaction_file, 'r') as f:
            next(f)
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    ligand_id = parts[0]
                    docking_score = parts[-1]
                    docking_scores.append({
                        "Ligand_id": ligand_id,
                        "Protein": protein,
                        "Docking_score": docking_score
                    })
    docking_score_df = pd.DataFrame(docking_scores)
    docking_score_df.to_csv(docking_score_file, index=False)
else:
    docking_score_df = pd.read_csv(docking_score_file)

merged_df = pd.merge(ligand_df, docking_score_df, on=['Protein', 'Ligand_id'], how="inner")
merged_df = merged_df.dropna()
merged_df = merged_df.drop_duplicates()
print(merged_df.head())

merged_df.to_csv(os.path.join(output_dir, 'ligand_with_docking_score.csv'), index=False)