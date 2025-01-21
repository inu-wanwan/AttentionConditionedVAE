import pandas as pd
import numpy as np
import argparse
import os
from utils import load_config

def select_min_error_from_bins(data, actual_col, predicted_col, bins=10):
    """
    Selects the value with the smallest error from each bin of the specified column.
    
    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        actual_col (str): Column name for actual docking scores.
        predicted_col (str): Column name for predicted docking scores.
        bins (int): The number of bins to divide the data into.

    Returns:
        pd.DataFrame: A DataFrame containing the row with the smallest error from each bin.
    """
    # Calculate error as the absolute difference between actual and predicted scores
    data['error'] = abs(data[actual_col] - data[predicted_col])

    # Bin the data based on the actual scores
    data['bin'] = pd.cut(data[actual_col], bins=bins, labels=False)

    # Select the row with the smallest error from each bin
    min_error_data = data.loc[data.groupby('bin')['error'].idxmin()].reset_index(drop=True)

    # Drop the temporary columns
    min_error_data = min_error_data.drop(columns=['bin', 'error'])
    
    return min_error_data

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model' , '-m', type=str, help='Model name e.g. schnet_ds_regression, ds_regression, etc.')
    parser.add_argument('--predictor_timestamp', '-p', type=str, required=True, help='Timestamp of the trained prediction model')
    parser.add_argument('--vae_timestamp', '-v', type=str, required=True, help='Timestamp of the trained VAE model')
    parser.add_argument('--result_file', '-r', type=str, default='results_test.csv', help='Name of the result file')
    args = parser.parse_args()

    # Load the config file
    config = load_config("filepath.yml")
    eval_dir = config['data']['eval']
    result_dir = os.path.join(eval_dir, args.model, args.predictor_timestamp)
    result_file = os.path.join(result_dir, args.result_file)
    output_dir = os.path.join(eval_dir, 'ds_regression', args.predictor_timestamp)

    # Load the prediction results
    results = pd.read_csv(result_file)

    # Sample from bins based on error distribution
    sampled_results = select_min_error_from_bins(results, 'Actual_Docking_Score', 'Predicted_Docking_Score', bins=10,)

    # Save the sampled results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'binwise_accurate.csv')
    sampled_results.to_csv(output_file, index=False)

    print(f"Sampled results saved to {output_file}")

if __name__ == "__main__":
    main()