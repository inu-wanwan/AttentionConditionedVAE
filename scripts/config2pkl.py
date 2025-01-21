import pickle
import os
from utils import load_config

file_config = load_config('filepath.yml')
model_config = load_config('model.yml')['docking_score_regression_model']

model_config['target'] = 'DRD3'

out_file = os.path.join(file_config['data']['model'], 'ds_regression', 'ds_2025-01-03_22-35-27', 'model_config.pkl')

with open(out_file, 'wb') as f:
    pickle.dump(model_config, f)

print(f"Model config saved at {out_file}")