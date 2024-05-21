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


def write_json(data, file_path):
    check_dir_exist('/'.join(file_path.split('/')[:-1]))
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def check_dir_exist(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
