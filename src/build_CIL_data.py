import argparse
import random

from utils import read_csv, set_seed, write_json


def get_data(path):
    train_data = read_csv(f'{path}/train.csv')
    val_data = read_csv(f'{path}/val.csv')
    test_data = read_csv(f'{path}/test.csv')
    return train_data, val_data, test_data


def get_data_dict(data):
    data_group = data.groupby('author_id')['text'].apply(list).reset_index()
    data_dict = dict(zip(data_group['author_id'], data_group['text']))
    # data_dict = [{'author_id': author_id, 'contents': contents} for author_id, contents in data_pack.items()]
    return data_dict


def get_sessions(train, val, test):
    author_ids = list(train.keys())

    random.shuffle(author_ids)

    total_num_authors = len(author_ids)

    slice_values = [0.5] + [0.1] * 5

    data_sessions = []
    for s_value in slice_values:
        slice = int(total_num_authors * s_value)
        selected_author_ids, author_ids = author_ids[:slice], author_ids[slice:]
        session = get_single_session(train, val, test, selected_author_ids)
        data_sessions.append(session)

    if author_ids:  # Add the remaining data into the last session
        remaining_data = get_single_session(train, val, test, author_ids)
        data_sessions[-1]['train'].update(remaining_data['train'])
        data_sessions[-1]['val'].update(remaining_data['val'])
        data_sessions[-1]['test'].update(remaining_data['test'])

    return data_sessions


def get_single_session(train, val, test, author_ids):
    selected_train = {key: value for key, value in train.items() if key in author_ids}
    selected_val = {key: value for key, value in val.items() if key in author_ids}
    selected_test = {key: value for key, value in test.items() if key in author_ids}
    return {'train': selected_train, 'val': selected_val, 'test': selected_test}


def write_data_sessions(data_sessions, dataset_name):
    for session_idx, session in enumerate(data_sessions):
        write_json(session['train'], f'../data/{dataset_name}_CIL/session_{session_idx}/train.json')
        write_json(session['val'], f'../data/{dataset_name}_CIL/session_{session_idx}/val.json')
        write_json(session['test'], f'../data/{dataset_name}_CIL/session_{session_idx}/test.json')


if __name__ == '__main__':
    set_seed(seed_num=42)  # Set the seed for reproducibility

    DATA2PATH = {'blog50': '../data/blog50', 'imdb62': '../data/imdb62'}

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--dataset', '-d', type=str, help='Dataset used for build CIL data',
                        choices=DATA2PATH.keys(), default='blog50')
    ARGS = PARSER.parse_args()

    RAW_TRAIN, RAW_VAL, RAW_TEST = get_data(DATA2PATH[ARGS.dataset])

    TRAIN_DATA = get_data_dict(RAW_TRAIN)
    VAL_DATA = get_data_dict(RAW_VAL)
    TEST_DATA = get_data_dict(RAW_TEST)

    DATA_SESSIONS = get_sessions(TRAIN_DATA, VAL_DATA, TEST_DATA)

    write_data_sessions(DATA_SESSIONS, ARGS.dataset)
