import pandas as pd
import random
import os
from utils import load_config

"""
split_dataset_by_clusters.py
Splits the dataset by clusters into training, test, and validation sets.
"""

path_config = load_config('filepath.yml')
train_dir = path_config['data']['train']
test_dir = path_config['data']['test']
val_dir = path_config['data']['val']

dataset_file = os.path.join(path_config['data']['preprocessed'], 'dataset_final.csv')
cluster_file = os.path.join(path_config['data']['protein'], 'clusters-by-entity-40.txt')

dataset = pd.read_csv(dataset_file)

# convert pdb id to upper case
dataset['PDB ID'] = dataset['PDB ID'].str.upper()

# load the clusters

with open(cluster_file, 'r') as f:
    clusters = [line.strip().split() for line in f]

cluster_dict = {}
for cluster_id, cluster in enumerate(clusters):
    for protein_id in cluster:
        parts = protein_id.split('_')
        if len(parts) == 2:
            pdb_id, entity_id = parts[0], parts[1]
            if entity_id == "1":
                cluster_dict[pdb_id] = cluster_id

dataset['ClusterID'] = dataset['PDB ID'].map(cluster_dict)
# split the dataset by clusters

train_set = []
val_set = []
test_set = []

for cluster_id, group in dataset.groupby('ClusterID'):
    rand_num = random.random()
    if rand_num < 0.8:
        train_set.append(group)
    elif rand_num < 0.9:
        val_set.append(group)
    else:
        test_set.append(group)
       

train_set = pd.concat(train_set, ignore_index=True)
val_set = pd.concat(val_set, ignore_index=True)
test_set = pd.concat(test_set, ignore_index=True)

# save the datasets
train_set.to_csv(os.path.join(train_dir, 'train.csv'), index=False)
val_set.to_csv(os.path.join(val_dir, 'val.csv'), index=False)
test_set.to_csv(os.path.join(test_dir, 'test.csv'), index=False)

print(f"Train set saved to {os.path.join(train_dir, 'train.csv')}")
print(f"Validation set saved to {os.path.join(val_dir, 'val.csv')}")
print(f"Test set saved to {os.path.join(test_dir, 'test.csv')}")