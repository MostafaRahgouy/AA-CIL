import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(file_path, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, output_prefix='output'):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Ensure the sum of ratios is 1
    assert train_ratio + val_ratio + test_ratio == 1, "The sum of train, val, and test ratios must be 1"

    # Generate unique integer ids for authors
    author_ids = {author: idx for idx, author in enumerate(df['author'].unique())}
    df['author_id'] = df['author'].map(author_ids)

    # Rename 'abstract' to 'text'
    df.rename(columns={'abstract': 'text'}, inplace=True)

    # Select only the necessary columns
    df = df[['author_id', 'text']]

    # Initialize empty dataframes for train, val, test
    train_df = pd.DataFrame(columns=['author_id', 'text'])
    val_df = pd.DataFrame(columns=['author_id', 'text'])
    test_df = pd.DataFrame(columns=['author_id', 'text'])

    # Split data for each author
    for author_id in df['author_id'].unique():
        author_data = df[df['author_id'] == author_id]
        train_val, test = train_test_split(author_data, test_size=test_ratio, random_state=42)
        train, val = train_test_split(train_val, test_size=val_ratio / (train_ratio + val_ratio), random_state=42)

        # Concatenate the results
        train_df = pd.concat([train_df, train])
        val_df = pd.concat([val_df, val])
        test_df = pd.concat([test_df, test])

    # Save to CSV files
    train_df.to_csv(f'{output_prefix}train.csv', index=False)
    val_df.to_csv(f'{output_prefix}val.csv', index=False)
    test_df.to_csv(f'{output_prefix}test.csv', index=False)

if __name__ == '__main__':
    split_dataset(file_path='../data/org/arxiv100/arXiv_100authors_comp_sci.csv', output_prefix='../data/org/arxiv100/')