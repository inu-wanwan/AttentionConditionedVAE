import torch
import os
import pandas as pd
from tqdm import tqdm
from utils import load_config
from src.data.dataloader import get_dataloader
from src.score_prediction_models.docking_score_predictor import DockingScorePredictor

def predict_scores(device, model, dataloader, output_file):
    """
    Predict docking scores using the model and dataloader, and save results to a CSV file.

    Args:
        device: torch.device, computation device (CPU or GPU).
        model: PyTorch model, the trained docking score predictor.
        dataloader: DataLoader, batched data loader for input data.
        output_file: str, path to save the results as a CSV file.
    """
    model.eval()
    model.to(device)

    results = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting docking scores"):
            # Load batch data
            smiles = batch['smiles']
            protein_id = batch['protein_id']
            docking_scores = batch['docking_score'].to(device)
            smiles_embedding = batch['smiles_embedding'].to(device)
            protein_embedding = batch['protein_embedding'].to(device)

            # Predict scores
            pred_scores = model(smiles_embedding, protein_embedding).cpu()

            # Collect results
            for i in range(len(smiles)):
                results.append({
                    'SMILES': smiles[i],
                    'Protein_ID': protein_id[i],
                    'Actual_Docking_Score': docking_scores[i].item(),
                    'Predicted_Docking_Score': pred_scores[i].item()
                })

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

def main():
    """
    Main function to evaluate the docking score prediction model using batch processing.
    """
    # Load configurations
    file_config = load_config('filepath.yml')
    data_config = load_config('data.yml')
    model_config = load_config('model.yml')['docking_score_regression_model']

    # File paths
    timestamp = '2024-12-28_22-17-43'
    model_dir = file_config['data']['docking']
    model_file = os.path.join(model_dir, f"ds_{timestamp}", 'model.pth')
    test_file = os.path.join(file_config['data']['test'], 'test_FNTA.csv')
    train_file = os.path.join(file_config['data']['train'], 'train_FNTA.csv')
    os.makedirs(os.path.join(file_config['data']['eval'], 'ds_regression', timestamp), exist_ok=True)
    results_file = os.path.join(file_config['data']['eval'], 'ds_regression', timestamp, 'results_test.csv')

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = DockingScorePredictor(
        embed_dim=model_config['embed_dim'],
        num_heads=model_config['num_heads'],
        ffn_hidden_dim=model_config['ffn_hidden_dim'],
        num_transformer_blocks=model_config['num_transformer_blocks']
    )
    model.load_state_dict(torch.load(model_file))

    # DataLoader setup with batch processing
    batch_size = 128  # Define batch size
    test_dataloader = get_dataloader(
        csv_file=test_file,
        smiles_max_len=data_config['dataset']['smiles_max_len'],
        protein_max_len=data_config['dataset']['protein_max_len'],
        batch_size=batch_size,
        shuffle=False
    )

    # Predict docking scores and save results
    predict_scores(device, model, test_dataloader, results_file)

if __name__ == '__main__':
    main()
