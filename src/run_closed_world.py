import argparse
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from transformers import logging

logging.set_verbosity_error()
from data_provider import ClosedDataProvider
from models import BaselineModel
from utils import set_seed, check_dir_exist, write_json, read_csv

if __name__ == '__main__':
    set_seed(seed_num=42)  # Set the seed for reproducibility

    DATA2PATH = {'blog50': '../data/org/blog50',
                 'imdb62': '../data/org/imdb62',
                 'ccat50': '../data/org/ccat50',
                 'blog1000': '../data/org/blog1000'
                 }

    # Config
    PARSER = argparse.ArgumentParser()

    PARSER.add_argument('--dataset', '-d', type=str, help='Select the dataset for your experiment',
                        choices=DATA2PATH.keys(), default='blog1000')

    PARSER.add_argument('--model_type', '-md', type=str, help='Select the baseline model',
                        choices=['FT'], default='FT')

    PARSER.add_argument('--n_epochs', '-e', type=int, help='number of epochs', default=5)

    PARSER.add_argument('--batch_size', '-b', type=int, help='batch size', default=32)

    PARSER.add_argument('--device', '-de', type=str, choices=['cpu', 'cuda'], default='cuda')

    PARSER.add_argument('--lr', '-lr', type=float, help='learning rate', default=2e-5)

    PARSER.add_argument('--demo', action='store_true',
                        help='Enable this option to perform a full round of experiments for the sanitize check, '
                             'using only two samples per session.')

    PARSER.add_argument('--out_dir', '-od', type=str, help='path of the output directory that save output information',
                        default='../output')

    PARSER.add_argument('--save_init_model', action='store_true',
                        help='Save the model after the first session, save time'
                             'if the the first session model is same among difference experiments')

    CONFIG = PARSER.parse_args()

    # +++++++++++++++++++++++++++++++++++  DEFINE OUTPUT PATHS ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    SAVE_DIR_PATH = f'{CONFIG.out_dir}/{CONFIG.dataset}'
    SAVE_DIR_PATH += '/Closed_w'

    MODEL_DIR_PATH = f'{SAVE_DIR_PATH}/trained_models'
    RESULT_DIR_PATH = f'{SAVE_DIR_PATH}/results'
    LOG_DIR_PATH = f'{SAVE_DIR_PATH}/log'
    check_dir_exist(MODEL_DIR_PATH)
    check_dir_exist(RESULT_DIR_PATH)
    check_dir_exist(LOG_DIR_PATH)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    if CONFIG.dataset == 'blog1000':
        TRAIN_DATA = read_csv(f'{DATA2PATH[CONFIG.dataset]}/train_closed.csv')
        VAL_DATA = read_csv(f'{DATA2PATH[CONFIG.dataset]}/val_closed.csv')
        TEST_DATA = read_csv(f'{DATA2PATH[CONFIG.dataset]}/test_closed.csv')

    else:
        TRAIN_DATA = read_csv(f'{DATA2PATH[CONFIG.dataset]}/train.csv')
        VAL_DATA = read_csv(f'{DATA2PATH[CONFIG.dataset]}/val.csv')
        TEST_DATA = read_csv(f'{DATA2PATH[CONFIG.dataset]}/test.csv')

    TRAIN_DATA.rename(columns={'text': 'content'}, inplace=True)
    VAL_DATA.rename(columns={'text': 'content'}, inplace=True)
    TEST_DATA.rename(columns={'text': 'content'}, inplace=True)

    TRAIN_DATA = TRAIN_DATA.to_dict(orient='records')
    VAL_DATA = VAL_DATA.to_dict(orient='records')
    TEST_DATA = TEST_DATA.to_dict(orient='records')

    MODEL = BaselineModel(baseline_type=CONFIG.model_type, n_epochs=CONFIG.n_epochs, device=CONFIG.device,
                          lr=CONFIG.lr, log_dir=LOG_DIR_PATH)

    # Convert data into torch dataset
    TRAIN_DATASET = ClosedDataProvider(data=TRAIN_DATA, tokenizer=MODEL.get_tokenizer())
    VAL_DATASET = ClosedDataProvider(data=VAL_DATA, tokenizer=MODEL.get_tokenizer())
    TEST_DATASET = ClosedDataProvider(data=TEST_DATA, tokenizer=MODEL.get_tokenizer())

    # Define torch dataloader for the datasets
    TRAIN_LOADER = DataLoader(TRAIN_DATASET, batch_size=CONFIG.batch_size, shuffle=True)
    VAL_LOADER = DataLoader(VAL_DATASET, batch_size=CONFIG.batch_size, shuffle=False)
    TEST_LOADER = DataLoader(TEST_DATASET, batch_size=CONFIG.batch_size, shuffle=False)

    # =============================== Train ================================================
    NUM_AUTHORS = len(set([item['author_id'] for item in TRAIN_DATA]))

    MODEL.add_head(num_new_authors=NUM_AUTHORS)
    CURRENT_BEST_MODEL, LOG_LOSSES = MODEL.train_session(session=1, train_loader=TRAIN_LOADER, val_loader=VAL_LOADER)
    write_json(LOG_LOSSES, f'{LOG_DIR_PATH}/log.json')

    # =============================== test a session ================================================

    CURRENT_PREDICTIONS, CURRENT_LABELS, CURRENT_TRUE_AUTHOR_IDES = MODEL.test(session=1, test_loader=TEST_LOADER)

    CURRENT_ACC = accuracy_score(CURRENT_LABELS, CURRENT_PREDICTIONS)

    RESULTS_OUTPUT = {'session': 1, 'accuracy': round(CURRENT_ACC, 4) * 100, 'num_author': NUM_AUTHORS,
                      'true_author_ides': CURRENT_TRUE_AUTHOR_IDES,
                      'pred_author_ides': CURRENT_PREDICTIONS
                      }

    # =============================== saving the results ================================================

    MODEL.save_model(filename=f'{MODEL_DIR_PATH}/model.pth')
    write_json(RESULTS_OUTPUT, f'{RESULT_DIR_PATH}/result.json')
