from analyzer import extract_acc, plot_incremental_learning

if __name__ == '__main__':
    BLOG50_ALL_ACC, NUM_BLOG50_AUTHORS_INC = extract_acc(dir_path='../output/blog50/CIL/10_sessions',
                                                         dataset='blog50', num_sessions=10)

    IMDB62_ALL_ACC, NUM_IMDB62_AUTHORS_INC = extract_acc(dir_path='../output/imdb62/CIL/10_sessions',
                                                         dataset='imdb62', num_sessions=10)

    BLOG1000_ALL_ACC, NUM_BLOG1000_AUTHORS_INC = extract_acc(dir_path='../output/blog1000/CIL/10_sessions',
                                                             dataset='blog1000', num_sessions=10)

    CCAT50_ALL_ACC, NUM_CCAT50_AUTHORS_INC = extract_acc(dir_path='../output/ccat50/CIL/10_sessions',
                                                             dataset='ccat50', num_sessions=10)

    Arxiv100_ALL_ACC, NUM_arxiv100_AUTHORS_INC = extract_acc(dir_path='../output/arxiv100/CIL/10_sessions',
                                                             dataset='arxiv100', num_sessions=10)

    ITEMS = [
        (NUM_BLOG50_AUTHORS_INC, BLOG50_ALL_ACC, 'Blog50'),
        (NUM_IMDB62_AUTHORS_INC, IMDB62_ALL_ACC, 'IMDB62'),
        (NUM_BLOG1000_AUTHORS_INC, BLOG1000_ALL_ACC, 'Blog1000'),
        (NUM_CCAT50_AUTHORS_INC, CCAT50_ALL_ACC, 'CCAT50'),
        (NUM_arxiv100_AUTHORS_INC, Arxiv100_ALL_ACC, 'Arxiv100')
    ]

    plot_incremental_learning(ITEMS)
