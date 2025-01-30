import argparse
import pandas as pd

def calc_dockable(dock_csv, generated_num=1000):
    """
    Calculate the number of dockable molecules in a CSV file.
    
    Args:
        dock_csv (str): Path to the CSV file with docking scores.
        generated_num (int): Number of generated molecules.
    
    Returns:
        int: Number of dockable molecules.
    """
    df = pd.read_csv(dock_csv)
    unique_docking_titles = df["title"].drop_duplicates().astype(str)
    dockable_num = len(unique_docking_titles)
    docking_ratio = dockable_num / generated_num
    print(f"Number of dockable molecules: {dockable_num}")

    return docking_ratio

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the number of dockable molecules in a CSV file")
    parser.add_argument("--dock_csv", "-c", type=str, help="Path to the input CSV file")
    parser.add_argument("--generated_num", "-g", type=int, default=1000, help="Number of generated molecules")
    args = parser.parse_args()

    docking_ratio = calc_dockable(args.dock_csv, args.generated_num)
    print(f"Docking ratio: {docking_ratio}")