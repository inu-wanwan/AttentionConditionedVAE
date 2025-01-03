import torch
import os
import pandas as pd
from utils import load_config
from src.data.dataloader import get_dataloader
from src.score_prediction_models.docking_score_predictor import DockingScorePredictor

def extract_attn_map(device, model, smiles_emb, protein_emb):
    """
    Extract attention maps using the model and embeddings, and save results to a CSV file.

    Args:
        device: torch.device, computation device (CPU or GPU).
        model: PyTorch model, the trained docking score predictor.
        smiles_emb: torch.Tensor, the SMILES embeddings.
        protein_emb: torch.Tensor, the protein embeddings.
    Returns:
        attn_map: torch.Tensor, the attention map
    """
    model.eval()
    model.to(device)

    # Predict scores

    with torch.no_grad():
        docking_score, self_attn_wts, cross_attn_wts = model(smiles_emb.to(device), protein_emb.to(device))

    # Save results to CSV
    attn_map_0_df = pd.DataFrame(cross_attn_wts[0].squeeze(0).cpu().numpy())
    attn_map_1_df = pd.DataFrame(cross_attn_wts[1].squeeze(0).cpu().numpy())
    attn_map_2_df = pd.DataFrame(cross_attn_wts[2].squeeze(0).cpu().numpy())
    
    return attn_map_0_df, attn_map_1_df, attn_map_2_df

def main():
    """
    Main function to extract attention maps using the docking score prediction model.
    """
    # Load configurations
    file_config = load_config('filepath.yml')
    data_config = load_config('data.yml')
    model_config = load_config('model.yml')['docking_score_regression_model']

    # File paths
    timestamp = '2024-12-28_22-17-43'
    model_dir = file_config['data']['docking']
    model_path = os.path.join(model_dir, f"ds_{timestamp}", "model.pth")
    result_dir = os.path.join(file_config['data']['eval'], 'ds_regression', timestamp)
    test_file = os.path.join(file_config['data']['test'], 'test_FNTA.csv')

    # Load model
    model = DockingScorePredictor(
        embed_dim=model_config['embed_dim'],
        num_heads=model_config['num_heads'],
        ffn_hidden_dim=model_config['ffn_hidden_dim'],
        num_transformer_blocks=model_config['num_transformer_blocks'],
        return_attn_wts=True
    )

    model.load_state_dict(torch.load(model_path))

    # Load data
    ligand_id = 'ZINC36063486'

    test_dataloader = get_dataloader(
        csv_file=test_file,
        smiles_max_len=data_config['dataset']['smiles_max_len'],
        protein_max_len=data_config['dataset']['protein_max_len'],
        batch_size=1,
        shuffle=False,
        ligand_id=ligand_id
    )

    data = next(iter(test_dataloader))

    smiles_emb = data['smiles_embedding']
    protein_emb = data['protein_embedding']

    # Extract attention maps
    attn_map_0_df, attn_map_1_df, attn_map_2_df = extract_attn_map(torch.device('cuda'), model, smiles_emb, protein_emb)

    # Save results to CSV
    os.makedirs(result_dir, exist_ok=True)
    outfile_0 = os.path.join(result_dir, f'attn_map_0_{ligand_id}.csv')
    outfile_1 = os.path.join(result_dir, f'attn_map_1_{ligand_id}.csv')
    outfile_2 = os.path.join(result_dir, f'attn_map_2_{ligand_id}.csv')

    attn_map_0_df.to_csv(outfile_0, index=False, header=False)
    attn_map_1_df.to_csv(outfile_1, index=False, header=False)
    attn_map_2_df.to_csv(outfile_2, index=False, header=False)

    print(f"Attention maps saved to {result_dir}")

if __name__ == '__main__':
    main()