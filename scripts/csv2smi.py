import pandas as pd
import argparse

def create_smi_file_with_custom_id(input_file, output_file):
    """
    Generate a .smi file with custom IDs in the format f"{SEED_Ligand_ID}_{index}".

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the .smi file.
    """
    # Load the dataset
    df = pd.read_csv(input_file)

    # Check if required columns exist
    if 'SEED_Ligand_ID' not in df.columns or 'Generated_SMILES' not in df.columns:
        raise ValueError("Input file must contain 'SEED_Ligand_ID' and 'Generated_SMILES' columns.")

    # Create a custom ID column
    df['Custom_ID'] = df['SEED_Ligand_ID'] + "_" + df.index.astype(str)

    # Prepare the .smi data (SMILES and Custom_ID)
    smi_data = df[['Generated_SMILES', 'Custom_ID']]

    # Save to a .smi file (tab-separated)
    smi_data.to_csv(output_file, sep='\t', index=False, header=False)
    print(f".smi file created at: {output_file}")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Generate a .smi file with custom IDs.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to save the .smi file.")

    args = parser.parse_args()

    # Run the function
    create_smi_file_with_custom_id(args.input, args.output)
