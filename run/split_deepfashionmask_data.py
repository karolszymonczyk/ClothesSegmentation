import os
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def split_data(data_dir_path: str, test_size=0.2) -> None:
    data_dir_path_object = Path(data_dir_path)
    data_files_paths = data_dir_path_object.glob("*.parquet")
    dfs = [pd.read_parquet(path, engine="pyarrow") for path in data_files_paths]
    full_dataset_df = pd.concat(dfs, ignore_index=True)

    train_df, test_df = train_test_split(
        full_dataset_df, test_size=test_size, random_state=42
    )

    train_path = os.path.join(data_dir_path, "train.parquet")
    save_dataframe_to_parquet(train_df, train_path)

    test_path = os.path.join(data_dir_path, "test.parquet")
    save_dataframe_to_parquet(test_df, test_path)


def save_dataframe_to_parquet(df: pd.DataFrame, save_path: str) -> None:
    df.to_parquet(save_path, engine="pyarrow")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run this script to split DeepFashionMask dataset into train and test subsets."
    )
    parser.add_argument(
        "-d", "--data_path", help="Path to data directory", required=True
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    data_path = args.data_path
    split_data(data_path)
