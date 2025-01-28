import yaml
import os
import numpy as np
from utils import load_config

def get_protein_length(protein):
    """
    Get the length of the protein sequence.
    """
    path_config = load_config('filepath.yml')
    alphafold_dir = path_config['data']['alphafold']
    dude_dir = path_config['data']['DUD-E']

    # load uniprot id
    with open(os.path.join(dude_dir, protein.lower(), 'uniprot.txt')) as f:
        lines = f.readlines()
        uniprot_id = lines[0].strip()

    # load protein sequence
    embedding = np.load(os.path.join(alphafold_dir, uniprot_id, 'structure.npy'))

    return len(embedding)
    

def make_configs():
    """
    Make config files for training and model.
    """

    # Load path config
    path_config = load_config('filepath.yml')

    protein_list = ['AKT1', 'AMPC', 'CP3A4', 'CXCR4', 'GCR']

    for protein in protein_list:
        os.makedirs(os.path.join(path_config['config'], 'decoder_only', protein), exist_ok=True)
        config_dir = os.path.join(path_config['config'], 'decoder_only', protein)

        # Get protein length
        protein_length = get_protein_length(protein)

        # Model config
        model_config = {
            "target": protein,
            "embed_dim": 384,
            "smiles_max_len": 100,
            "protein_max_len": protein_length + 20,
            "transformer_layer_used" : 0,
            "moses_vae_file": "smiles_vae_dmqp1m_no_dot_dup.pt",
            "docking_score_regression": "ds_2025-01-17_16-05-23",
            "train_file": f"train_{protein}.csv",
            "val_file": f"val_{protein}.csv",
        }

        # Train config
        train_config = {
            "train_file": f"train_{protein}.csv",
            "val_file": f"val_{protein}.csv",
            "batch_size": 128,
            "epochs": 20,
            "lr": 1e-4,
            "save_freq": 2,
        }

        # Save model config
        model_config_path = os.path.join(config_dir, 'model.yml')
        train_config_path = os.path.join(config_dir, 'train.yml')

        with open(model_config_path, 'w') as f:
            yaml.dump(model_config, f)

        with open(train_config_path, 'w') as f:
            yaml.dump(train_config, f)

        print(f"Model config saved at {model_config_path}")
        print(f"Train config saved at {train_config_path}")

if __name__ == "__main__":
    make_configs()