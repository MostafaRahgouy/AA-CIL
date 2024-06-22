import argparse
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from transformers import logging

logging.set_verbosity_error()

from models import WeightRegularize
from data_provider import BaseDataProvider, RandomExemplarProvider, HerdingExemplarProvider, HardExemplarProvider
from utils import set_seed, read_data_sessions, check_dir_exist, write_json

if __name__ == '__main__':
    set_seed(seed_num=42)  # Set the seed for reproducibility

    DATA2PATH = {'blog50': '../data/CIL/blog50_CIL', 'imdb62': '../data/CIL/imdb62_CIL'}

    # Config
    PARSER = argparse.ArgumentParser()

    PARSER.add_argument('--dataset', '-d', type=str, help='Select the dataset for your experiment',
                        choices=DATA2PATH.keys(), default='blog50')

    PARSER.add_argument('--model_type', '-md', type=str, help='Select the baseline model',
                        choices=['EWC', 'MAS'], default='MAS')

    PARSER.add_argument('--n_epochs', '-e', type=int, help='number of epochs', default=5)

    PARSER.add_argument('--batch_size', '-b', type=int, help='batch size', default=32)

    PARSER.add_argument('--device', '-de', type=str, choices=['cpu', 'cuda'], default='cuda')

    PARSER.add_argument('--lr', '-lr', type=float, help='learning rate', default=2e-5)

    PARSER.add_argument('--demo', action='store_true',
                        help='Enable this option to perform a full round of experiments for the sanitize check, '
                             'using only two samples per session.')

    PARSER.add_argument('--few_shot', action='store_true',
                        help='If you wish to do the experiment with few_shot select this')

    PARSER.add_argument('--out_dir', '-od', type=str, help='path of the output directory that save output information',
                        default='../output')

    PARSER.add_argument('--init_model', '-im', type=str, help='Path of the initial model', default='')

    PARSER.add_argument('--save_init_model', action='store_true',
                        help='Save the model after the first session, save time'
                             'if the the first session model is same among difference experiments')

    CONFIG = PARSER.parse_args()

    # +++++++++++++++++++++++++++++++++++  DEFINE OUTPUT PATHS ++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    SAVE_DIR_PATH = f'{CONFIG.out_dir}/{CONFIG.dataset}'
    if CONFIG.few_shot:
        SAVE_DIR_PATH += '/FSCIL'
    else:
        SAVE_DIR_PATH += '/CIL'

    SAVE_DIR_PATH += f'/{CONFIG.model_type}'

    MODEL_DIR_PATH = f'{SAVE_DIR_PATH}/trained_models'
    RESULT_DIR_PATH = f'{SAVE_DIR_PATH}/results'
    LOG_DIR_PATH = f'{SAVE_DIR_PATH}/log'
    check_dir_exist(MODEL_DIR_PATH)
    check_dir_exist(RESULT_DIR_PATH)
    check_dir_exist(LOG_DIR_PATH)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    SELECTED_TRAIN_EXEMPLARS, SELECTED_VAL_EXEMPLARS = [], []

    DATA_SESSIONS, MAPPING_IDES, AUTHORS_PARTITION_CONFIG = read_data_sessions(DATA2PATH[CONFIG.dataset])

    MODEL = WeightRegularize(model_type=CONFIG.model_type, n_epochs=CONFIG.n_epochs, device=CONFIG.device,
                             lr=CONFIG.lr, log_dir=LOG_DIR_PATH)

    # Train and Test Sessions
    for SESSION, SESSION_DATA in enumerate(DATA_SESSIONS):

        # ========================================= CHECK Few-Shot =====================================================

        # if few-shot is selected, from second sessions just randomly take 10 samples for each author
        if SESSION != 0 and CONFIG.few_shot:

            CURRENT_TRAIN_DATA, CURRENT_VAL_DATA = RandomExemplarProvider().get_random_exemplars_set(
                current_data_session=SESSION_DATA, num_exp_per_author=10)
            CURRENT_TEST_DATA = [item for sublist in [data['test'] for data in DATA_SESSIONS[:SESSION]] for item in
                                 sublist] + SESSION_DATA['test']
        else:
            CURRENT_TRAIN_DATA = SESSION_DATA['train']
            CURRENT_VAL_DATA = SESSION_DATA['val']
            CURRENT_TEST_DATA = [item for sublist in [data['test'] for data in DATA_SESSIONS[:SESSION]] for item in
                                 sublist] + SESSION_DATA['test']

        # number of new authors in this session
        NUM_NEW_AUTHORS = AUTHORS_PARTITION_CONFIG[f'session_{SESSION}']

        if CONFIG.demo:  # Just for checking everything work properly or not
            CURRENT_TRAIN_DATA = CURRENT_TRAIN_DATA[:16]
            CURRENT_VAL_DATA = CURRENT_VAL_DATA[:16]
            CURRENT_TEST_DATA = CURRENT_TEST_DATA[:16]

        # Convert data into torch dataset
        TRAIN_DATASET = BaseDataProvider(data=CURRENT_TRAIN_DATA, tokenizer=MODEL.get_tokenizer())
        VAL_DATASET = BaseDataProvider(data=CURRENT_VAL_DATA, tokenizer=MODEL.get_tokenizer())
        TEST_DATASET = BaseDataProvider(data=CURRENT_TEST_DATA, tokenizer=MODEL.get_tokenizer())

        # Define torch dataloader for the datasets
        TRAIN_LOADER = DataLoader(TRAIN_DATASET, batch_size=CONFIG.batch_size, shuffle=True)
        VAL_LOADER = DataLoader(VAL_DATASET, batch_size=CONFIG.batch_size, shuffle=False)
        TEST_LOADER = DataLoader(TEST_DATASET, batch_size=CONFIG.batch_size, shuffle=False)

        # =============================== train a session ================================================

        if CONFIG.init_model and SESSION == 0:
            MODEL.custom_load_model(filename=CONFIG.init_model,
                                    train_loader=TRAIN_LOADER)  # load the model and compute the importance

        else:
            # Add a new head/classifier for the new authors introduced in this session
            MODEL.add_head(num_new_authors=NUM_NEW_AUTHORS)
            CURRENT_BEST_MODEL, LOG_LOSSES = MODEL.train_session(SESSION, TRAIN_LOADER, VAL_LOADER)
            write_json(LOG_LOSSES, f'{LOG_DIR_PATH}/{SESSION}_session_log.json')

        # =============================== test a session ================================================

        CURRENT_PREDICTIONS, CURRENT_LABELS, CURRENT_TRUE_AUTHOR_IDES = MODEL.test(SESSION, TEST_LOADER)

        CURRENT_ACC = accuracy_score(CURRENT_LABELS, CURRENT_PREDICTIONS)

        RESULTS_OUTPUT = {'session': SESSION, 'accuracy': round(CURRENT_ACC, 4) * 100, 'num_author': NUM_NEW_AUTHORS,
                          'true_author_ides': CURRENT_TRUE_AUTHOR_IDES,
                          'pred_author_ides': [MAPPING_IDES['inc_id_2_author_id'][str(item)] for item in
                                               CURRENT_PREDICTIONS],
                          'true_inc_ides': CURRENT_LABELS,
                          'pred_inc_ides': CURRENT_PREDICTIONS
                          }

        # =============================== saving the results ================================================

        if SESSION == 0 and CONFIG.save_init_model:
            MODEL.save_model(filename=f'{MODEL_DIR_PATH}/{SESSION}_session_model.pth')

        if SESSION == len(DATA_SESSIONS) - 1:
            MODEL.save_model(filename=f'{MODEL_DIR_PATH}/{SESSION}_session_model.pth')

        write_json(RESULTS_OUTPUT, f'{RESULT_DIR_PATH}/{SESSION}_session_result.json')
