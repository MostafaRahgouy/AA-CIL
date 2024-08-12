import pandas as pd
from utils import set_seed, check_dir_exist, write_json, read_csv



def remap_author_ids(train, val, test, output):

    # Combine the dataframes to create a consistent author mapping
    combined_df = pd.concat([train[['author_id']], test[['author_id']], val[['author_id']]])
    combined_df = combined_df.drop_duplicates().reset_index(drop=True)

    # Create a mapping from original author_id to new IDs starting from zero
    author_map = {author_id: idx for idx, author_id in enumerate(combined_df['author_id'].unique())}

    # Apply the mapping to each dataframe
    train['author_id'] = train['author_id'].map(author_map)
    test['author_id'] = test['author_id'].map(author_map)
    val['author_id'] = val['author_id'].map(author_map)

    train.to_csv(f'{output}/train_closed.csv', index=False)
    val.to_csv(f'{output}/val_closed.csv', index=False)
    test.to_csv(f'{output}/test_closed.csv', index=False)



if __name__ == '__main__':
    TRAIN_DATA = read_csv(f'../data/org/blog1000/train.csv')
    VAL_DATA = read_csv(f'../data/org/blog1000/val.csv')
    TEST_DATA = read_csv(f'../data/org/blog1000/test.csv')

    remap_author_ids(train=TRAIN_DATA, val=VAL_DATA, test=TEST_DATA, output='../data/org/blog1000')
