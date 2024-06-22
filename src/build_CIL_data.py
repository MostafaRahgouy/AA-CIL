"""
Build_CIL_data script: This script processes the original data in the following format:

    Input data: a list of samples where each sample contains an AUTHOR_ID alongside CONTENT. The AUTHOR_ID is a unique
    identifier of an author, and the CONTENT record is their text.

It builds data suitable for Class-incremental Learning based on the following algorithm:

    1. Group content by each unique AUTHOR_ID into a dictionary, e.g., {author_id: [content1, content2, ...], ...}.
    2. Randomly select a set of authors along with all texts associated with them for each session.
    3. Assign incremental indexes to authors, which will be used to train the models.
    4. Flatten the obtained dictionary into a list of samples with AUTHOR_ID, INCREMENTAL_ID, and CONTENT.

The output data should be in the following format:

    AUTHOR_ID: the original unique author id
    INCREMENTAL_ID: the assigned id in an incremental way
    CONTENT: a single text associated with the author
"""

import argparse
import copy
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
    return data_dict


def add_incremental_ides_to_data_sessions(data_sessions, mapping_ides):
    incremental_idx = 0
    for session in data_sessions:
        add_incremental_ides(session['train'], incremental_idx, mapping_ides)
        add_incremental_ides(session['val'], incremental_idx, mapping_ides)
        add_incremental_ides(session['test'], incremental_idx, mapping_ides)
        incremental_idx += len(session['train'])


def add_incremental_ides(authors, current_idx, mapping_ides):
    for author in authors:
        author.update({'incremental_id': current_idx})
        mapping_ides['author_id_2_inc_id'].update({author['author_id']: author['incremental_id']})
        mapping_ides['inc_id_2_author_id'].update({author['incremental_id']: author['author_id']})
        current_idx += 1


def flatten_authors_samples(data_session):
    flatten_data_sessions = []

    for session in data_session:
        train_flatten_session = flatten_single_author_samples(session['train'])
        val_flatten_session = flatten_single_author_samples(session['val'])
        test_flatten_session = flatten_single_author_samples(session['test'])

        flatten_data_sessions.append({'train': train_flatten_session,
                                      'val': val_flatten_session,
                                      'test': test_flatten_session})
    return flatten_data_sessions


def flatten_single_author_samples(authors):
    flatten_session = []
    for author in authors:
        author_samples = []
        for content in author['contents']:
            author_samples.append({'author_id': author['author_id'],
                                   'incremental_id': author['incremental_id'],
                                   'content': content})
        flatten_session += author_samples
    random.shuffle(flatten_session)
    return flatten_session


def get_single_session(train, val, test, author_ids):
    selected_train = [{'author_id': key, 'contents': value} for key, value in train.items() if key in author_ids]
    selected_val = [{'author_id': key, 'contents': value} for key, value in val.items() if key in author_ids]
    selected_test = [{'author_id': key, 'contents': value} for key, value in test.items() if key in author_ids]
    return {'train': selected_train, 'val': selected_val, 'test': selected_test}


def write_data_sessions(data_sessions, dataset_name):
    for session_idx, session in enumerate(data_sessions):
        write_json(session['train'], f'../data/CIL/{dataset_name}_CIL/session_{session_idx}/train.json')
        write_json(session['val'], f'../data/CIL/{dataset_name}_CIL/session_{session_idx}/val.json')
        write_json(session['test'], f'../data/CIL/{dataset_name}_CIL/session_{session_idx}/test.json')


def prepare_histogram_data(data_session):
    histogram_data = []
    for session in data_session:
        temp_data = []
        for train_author, val_author, test_author in zip(session['train'], session['val'], session['test']):
            if train_author['author_id'] != val_author['author_id'] or train_author['author_id'] != test_author[
                'author_id']:
                raise ValueError('unexpected error happened')
            temp_data.append(
                {f"a_{train_author['author_id']}": len(
                    train_author['contents'] + val_author['contents'] + test_author['contents'])})
        random.shuffle(temp_data)
        histogram_data.append(temp_data)
    return histogram_data


def get_sessions(train, val, test):
    mapping_ides = {'author_id_2_inc_id': {},
                    'inc_id_2_author_id': {}}  # this dictionary is going to use for map the original author ides to the incremental ides

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
        data_sessions[-1]['train'] += remaining_data['train']
        data_sessions[-1]['val'] += remaining_data['val']
        data_sessions[-1]['test'] += remaining_data['test']

    hist_data = prepare_histogram_data(data_sessions)

    author_size_for_sessions = {f'session_{session}': len(session_data['train']) for session, session_data in
                                enumerate(data_sessions)}
    add_incremental_ides_to_data_sessions(data_sessions, mapping_ides)

    flatten_data_sessions = flatten_authors_samples(data_sessions)

    return flatten_data_sessions, mapping_ides, author_size_for_sessions, hist_data


if __name__ == '__main__':
    set_seed(seed_num=42)  # Set the seed for reproducibility

    DATA2PATH = {'blog50': '../data/org/blog50', 'imdb62': '../data/org/imdb62'}

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--dataset', '-d', type=str, help='Dataset used for build CIL data',
                        choices=DATA2PATH.keys(), default='blog50')
    ARGS = PARSER.parse_args()

    RAW_TRAIN, RAW_VAL, RAW_TEST = get_data(DATA2PATH[ARGS.dataset])

    TRAIN_DATA = get_data_dict(RAW_TRAIN)
    VAL_DATA = get_data_dict(RAW_VAL)
    TEST_DATA = get_data_dict(RAW_TEST)

    DATA_SESSIONS, MAPPING_IDES, AUTHOR_SIZE_FOR_SESSIONS, HIST_DATA = get_sessions(TRAIN_DATA, VAL_DATA, TEST_DATA)

    write_data_sessions(DATA_SESSIONS, ARGS.dataset)
    write_json(MAPPING_IDES, f'../data/CIL/{ARGS.dataset}_CIL/mapping_ides.json')
    write_json(AUTHOR_SIZE_FOR_SESSIONS, f'../data/CIL/{ARGS.dataset}_CIL/authors_partition_config.json')
    write_json(HIST_DATA, f'../analysis/hist_data/{ARGS.dataset}.json')
