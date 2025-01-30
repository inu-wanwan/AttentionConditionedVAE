import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def plot_histogram(df, column_name, outfile, bins=30, density=False, color='blue', alpha=0.6, title=None, xlabel=None, ylabel=None):
    """
    指定したデータフレームのカラムのヒストグラムを作成する関数。

    Args:
        df (pd.DataFrame): ヒストグラムを作成するデータフレーム。
        column_name (str): ヒストグラムを作成する対象のカラム名。
        bins (int, optional): ビンの数。デフォルトは30。
        density (bool, optional): True の場合、確率密度として表示（y軸が正規化される）。False の場合、頻度（カウント）を表示。デフォルトは False。
        color (str, optional): ヒストグラムの色。デフォルトは 'blue'。
        alpha (float, optional): ヒストグラムの透明度。デフォルトは 0.6。
        title (str, optional): ヒストグラムのタイトル。指定がない場合はデフォルトでカラム名を使用。
        xlabel (str, optional): X軸のラベル。指定がない場合はカラム名を使用。
        ylabel (str, optional): Y軸のラベル。density=True の場合は "Density"、False の場合は "Count"。

    Returns:
        None: ヒストグラムを表示する。
    """
    active_df = df[df['Active'] == True]
    plt.figure(figsize=(8, 6))
    bins = int(bins)

    if density:
        sns.kdeplot(df[column_name], color=color, linestyle='--', linewidth=1.5, label='All')
        sns.kdeplot(active_df[column_name], color='orange', linestyle='--', linewidth=1.5, label='Active')
    else:
        sns.histplot(df[column_name], bins=bins, color=color, alpha=alpha, label='All')
        sns.histplot(active_df[column_name], bins=bins, color='red', alpha=alpha, label='Active')

    plt.legend()

    # ラベルとタイトルの設定
    plt.xlabel(xlabel if xlabel else column_name, fontsize=14)
    plt.ylabel(ylabel if ylabel else ("Density" if density else "Count"), fontsize=14)
    plt.title(title if title else f"Histogram of {column_name}", fontsize=16)
    
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.savefig(outfile, dpi=600)
    print(f"Saved histogram to {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot histogram of a column in a CSV file")
    parser.add_argument("--csv_file", "-c", type=str, help="Path to the input CSV file")
    parser.add_argument("--column_name", "-n", type=str, help="Name of the column to plot")
    parser.add_argument("--outfile", "-o", type=str, help="Path to the output file")
    parser.add_argument("--bins", "-b", type=int, default=30, help="Number of bins in the histogram")
    parser.add_argument("--density", "-d", action="store_true", help="Plot the histogram as a probability density")
    parser.add_argument("--color", "-clr", type=str, default="blue", help="Color of the histogram")
    parser.add_argument("--alpha", "-a", type=float, default=0.6, help="Transparency of the histogram")
    parser.add_argument("--title", "-t", type=str, help="Title of the histogram")
    parser.add_argument("--xlabel", "-xl", type=str, help="Label of the x-axis")
    parser.add_argument("--ylabel", "-yl", type=str, help="Label of the y-axis")
    args = parser.parse_args()

    # Load the CSV file
    df = pd.read_csv(args.csv_file)

    print(args.bins)

    # Plot the histogram
    plot_histogram(df=df, column_name=args.column_name, outfile=args.outfile, bins=args.bins, density=args.density, color=args.color, alpha=args.alpha, title=args.title, xlabel=args.xlabel, ylabel=args.ylabel)