import os
import json
import torch
import random
import numpy as np
import pandas as pd


def read_csv(file_path):
    try:
        dataframe = pd.read_csv(file_path, encoding='cp1252')
    except UnicodeDecodeError:
        dataframe = pd.read_csv(file_path, encoding='utf-8')
    return dataframe


def set_seed(seed_num=42):
    random_seed = seed_num
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    torch.backends.cudnn.deterministic = True


def write_json(data, file_path):
    check_dir_exist('/'.join(file_path.split('/')[:-1]))
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def read_data_sessions(dataset_dir, num_sessions=6):
    data_sessions = []
    for session in range(num_sessions):
        train = read_single_session(path=f'{dataset_dir}/session_{session}/train.json')
        val = read_single_session(path=f'{dataset_dir}/session_{session}/val.json')
        test = read_single_session(path=f'{dataset_dir}/session_{session}/test.json')

        data_sessions.append({'train': train, 'val': val, 'test': test})

    mapping_ides = read_json(f'{dataset_dir}/mapping_ides.json')
    authors_partition_config = read_json(f'{dataset_dir}/authors_partition_config.json')

    return data_sessions, mapping_ides, authors_partition_config


def read_single_session(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def check_dir_exist(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
