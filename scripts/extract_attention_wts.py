import os
import torch
import yaml
from utils import load_config
from src.data.dataloader import get_dataloader
from src.score_prediction_models.docking_score_predictor import DockingScorePredictor


# load config
file_config = load_config('filepath.yml')
data_config = load_config('data.yml')

# model
model_dir = os.path.join(file_config['data']['model'], 'ds_regression', 'docking_score_regression_model_2024-12-20_18-34-06')
model_config_file = os.path.join(model_dir, 'config.yaml')
with open(model_config_file, 'r') as ymlfile:
    model_config = yaml.safe_load(ymlfile)

model_config = model_config['model_config']['value']

# file paths
test_data_file = os.path.join(file_config['data']['test'], 'test.csv')
model_file = os.path.join(model_dir, 'model.pth')


# model
model = DockingScorePredictor(
    embed_dim=model_config['embed_dim'],
    num_heads=model_config['num_heads'],
    ffn_hidden_dim=model_config['ffn_hidden_dim'],
    num_transformer_blocks=model_config['num_transformer_layers'],
    return_attn_wts=True
).cuda()

# load model
model.load_state_dict(torch.load(model_file, weights_only=True))
model.eval()

# dataloader
test_dataloader = get_dataloader(
    csv_file=test_data_file,
    smiles_max_len=data_config['dataset']['smiles_max_len'],
    protein_max_len=data_config['dataset']['protein_max_len'],
    batch_size=1,
    shuffle=False,
)

with torch.no_grad():
    for batch in test_dataloader:
        smiles_embedding = batch['smiles_embedding'].cuda()
        af2_embedding = batch['protein_embedding'].cuda()
        docking_score = batch['docking_score'].cuda()
        docking_score_pred, self_attn_wts_list, cross_attn_wts_list = model(smiles_embedding, af2_embedding)
        print(f"dock_score: {docking_score.item()}")
        print(f"dock_score_pred: {docking_score_pred.item()}")
        print(f"self_attn_wts_list_len: {len(self_attn_wts_list)}")
        print(f"cross_attn_wts_list_len: {len(cross_attn_wts_list)}")
        print(f"cross_attn_wts_list[0].shape: {cross_attn_wts_list[0].shape}")
        break