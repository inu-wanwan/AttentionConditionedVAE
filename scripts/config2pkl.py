import pickle
import os
from utils import load_config

file_config = load_config('filepath.yml')
model_config = {
    "docking_score_regression": "ds_2025-01-03_22-35-27",
    "embed_dim": 384,
    "moses_vae_file": "smiles_vae_dmqp1m_no_dot_dup.pt",
    "protein_max_len": 400,
    "smiles_max_len": 100,
    "target": "DRD3",
    "train_file": "train_DRD3.csv",
    "transformer_layer_used": 0,
    "val_file": "val_DUD3.csv",
    }
model_config['target'] = 'DRD3'

out_file = os.path.join(file_config['data']['model'], 'decoder_only_model', '2025-01-10_20-12-20', 'model_config.pkl')

with open(out_file, 'wb') as f:
    pickle.dump(model_config, f)

print(f"Model config saved at {out_file}")