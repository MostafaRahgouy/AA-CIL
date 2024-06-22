from analyzer import plot_histogram, extract_acc, plot_incremental_learning
from utils import read_json

if __name__ == '__main__':
    IMDB62_HIST_DATA = read_json('../analysis/hist_data/imdb62.json')
    BLOG50_HIST_DATA = read_json('../analysis/hist_data/blog50.json')

    # plot_histogram({'IMDB62': IMDB62_HIST_DATA}, {'BLOG50': BLOG50_HIST_DATA})

    BLOG50_ALL_ACC, NUM_BLOG50_AUTHORS_INC  = extract_acc(dir_path='../output/blog50/CIL/', dataset='blog50')
    IMDB62_ALL_ACC, NUM_IMDB62_AUTHORS_INC = extract_acc(dir_path='../output/imdb62/CIL/', dataset='imdb62')
    plot_incremental_learning(NUM_BLOG50_AUTHORS_INC, BLOG50_ALL_ACC, 'BLOG50',
                              NUM_IMDB62_AUTHORS_INC, IMDB62_ALL_ACC, 'IMDB62')
