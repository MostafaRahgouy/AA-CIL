from utils import read_csv
import pandas as pd


def split_raw_data(data_df):
    # Split the samples of each author into train, val, and test sets
    train_list = []
    val_list = []
    test_list = []

    # Define the ratios
    train_ratio = 0.6
    val_ratio = 0.2

    # Group by author_id
    grouped = data_df.groupby('author_id')

    for author, group in grouped:
        # Shuffle the samples of the current author
        group = group.sample(frac=1, random_state=42).reset_index(drop=True)

        # Compute the number of samples for each split
        n_samples = len(group)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)

        # Split the samples
        train_samples = group.iloc[:n_train]
        val_samples = group.iloc[n_train:n_train + n_val]
        test_samples = group.iloc[n_train + n_val:]

        # Append to the respective lists
        train_list.append(train_samples)
        val_list.append(val_samples)
        test_list.append(test_samples)

    # Concatenate the lists to form the final DataFrames
    train_df = pd.concat(train_list).reset_index(drop=True)
    val_df = pd.concat(val_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)
    return train_df, val_df, test_df


if __name__ == '__main__':
    RAW_TRAIN = read_csv('../data/org/ccat50/raw_train.csv')
    RAW_VAL = read_csv('../data/org/ccat50/raw_val.csv')
    RAW_TEST = read_csv('../data/org/ccat50/raw_test.csv')

    ALL_DATA = pd.concat([RAW_TRAIN, RAW_VAL, RAW_TEST])

    TRAIN_DF, VAL_DF, TEST_DF = split_raw_data(ALL_DATA)

    TRAIN_DF.to_csv('../data/org/ccat50/train.csv')
    VAL_DF.to_csv('../data/org/ccat50/val.csv')
    TEST_DF.to_csv('../data/org/ccat50/test.csv')
