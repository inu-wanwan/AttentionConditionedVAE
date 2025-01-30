import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import argparse

def plot_docking_scores_with_seed(generated_data, docking_data, out_file,
                                      actual_score_column='r_i_docking_score',
                                      title_column='title', 
                                      seed_ligand_column='SEED_Ligand_ID', 
                                      docking_score_column='Docking_score',
                                      score_threshold=(-15, 0),
                                      num_compounds=None):
    """
    Plots boxplots for docking scores of generated compounds per seed compound,
    filtering out invalid or extreme docking scores based on a specified threshold,
    retaining only the best (lowest) docking score for each unique title,
    and allowing selection of a specified number of compounds with minimal score bias.

    Args:
        generated_data (DataFrame): DataFrame containing generated compounds and their docking scores.
        docking_data (DataFrame): DataFrame containing docking results.
        actual_score_column (str): Column in docking_data with actual docking scores.
        title_column (str): Column in docking_data containing titles to link to generated compounds.
        seed_ligand_column (str): Column in generated_data containing seed ligand IDs.
        docking_score_column (str): Column in generated_data with docking scores.
        score_threshold (tuple): Min and max values for valid docking scores (default: (-15, 0)).
        num_compounds (int, optional): Number of seed compounds to display in the plot. Default is None (all compounds).

    Returns:
        None
    """
    # Extract seed ligand ID from title (removing the generated number part)
    docking_data['Seed_Ligand_ID'] = docking_data[title_column].apply(lambda x: re.sub(r'_\d+$', '', x))
    
    # Merge generated data with docking data to include seed ligand information
    merged_data = docking_data.merge(generated_data[[seed_ligand_column, docking_score_column]], 
                                     left_on='Seed_Ligand_ID', right_on=seed_ligand_column, how='left')
    
    # Filter out invalid docking scores
    valid_data = merged_data[
        (merged_data[actual_score_column] >= score_threshold[0]) & 
        (merged_data[actual_score_column] <= score_threshold[1])
    ]
    
    # Retain only the best (lowest) docking score for each unique title
    best_data = valid_data.loc[
        valid_data.groupby(title_column)[actual_score_column].idxmin()
    ]
    
    # Sort seed ligands by their corresponding docking scores
    sorted_seed_data = best_data.drop_duplicates(subset=seed_ligand_column).sort_values(by=docking_score_column)
    
    # Select a subset of seed ligands with minimal bias in score distribution
    if num_compounds and num_compounds < len(sorted_seed_data):
        quantiles = np.linspace(0, 1, num_compounds + 2)[1:-1]  # Generate evenly spaced quantiles
        selected_indices = (sorted_seed_data[docking_score_column]
                            .quantile(quantiles)
                            .index)
        sorted_seed_data = sorted_seed_data.loc[selected_indices]
    
    sorted_seed_ligand_ids = sorted_seed_data[seed_ligand_column].unique()
    
    fig, ax = plt.subplots(figsize=(12, 6))

    # Collect data for boxplots
    boxplot_data = []
    labels = []
    seed_scores = []

    for seed_ligand in sorted_seed_ligand_ids:
        # Filter generated compounds for this seed ligand
        filtered_generated = best_data[
            best_data[seed_ligand_column] == seed_ligand
        ]

        if not filtered_generated.empty:
            # Add the best scores to the boxplot data
            boxplot_data.append(filtered_generated[actual_score_column].values)
            labels.append(seed_ligand)

            # Find the seed docking score from sorted data
            seed_score = sorted_seed_data[
                sorted_seed_data[seed_ligand_column] == seed_ligand
            ][docking_score_column].values[0]
            seed_scores.append(seed_score)

    # Plot boxplots
    box = ax.boxplot(boxplot_data, vert=True, patch_artist=True, labels=labels)
    ax.set_title("Filtered Best Docking Scores Per Seed Compound")
    ax.set_ylabel("Docking Score")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Overlay seed compound docking scores as red points
    for i, score in enumerate(seed_scores):
        ax.scatter([i + 1], [score], color='red', label="Seed Compound" if i == 0 else "", zorder=5)

    # Add legend
    ax.legend()

    # Adjust layout and show plot
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_file, dpi=600)
    print(f"Saved boxplot to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot boxplots for docking scores of generated compounds per seed compound")
    parser.add_argument("--generated_csv", "-g", type=str, help="Path to the generated CSV file")
    parser.add_argument("--docked_csv", "-d", type=str, help="Path to the generated CSV file")
    parser.add_argument("--out_file", "-o", type=str, help="Path to the output file")
    parser.add_argument("--actual_score_column", "-a", type=str, default="r_i_docking_score", help="Column in generated data with actual docking scores")
    parser.add_argument("--title_column", "-t", type=str, default="title", help="Column in generated data containing titles to link to seed compounds")
    parser.add_argument("--ligand_id_column", "-l", type=str, default="Ligand_id", help="Column in seed data containing ligand IDs")
    parser.add_argument("--docking_score_column", "-dclm", type=str, default="Docking_score", help="Column in seed data with docking scores")
    parser.add_argument("--score_threshold", "-st", type=float, nargs=2, default=(-15, 0), help="Min and max values for valid docking scores")
    parser.add_argument("--num_compounds", "-n", type=int, help="Number of seed compounds to display in the plot")
    args = parser.parse_args()


    generated_data = pd.read_csv(args.generated_csv)
    docking_data = pd.read_csv(args.docked_csv) 

    plot_docking_scores_with_seed(generated_data=generated_data,
                                  docking_data=docking_data,
                                  out_file=args.out_file,
                                  actual_score_column=args.actual_score_column,
                                  title_column=args.title_column,
                                  seed_ligand_column=args.ligand_id_column,
                                  docking_score_column=args.docking_score_column,
                                  score_threshold=args.score_threshold,
                                  num_compounds=args.num_compounds)