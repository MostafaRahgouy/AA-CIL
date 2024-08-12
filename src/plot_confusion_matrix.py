import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import read_json
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


def plot_heatmap(confusion_data):
    num_plots = len(confusion_data)
    num_cols = 5
    num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate the number of rows needed
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))

    # Flatten axes array if there is more than one row
    if num_rows > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, (predictions, labels, num_author, name) in enumerate(tqdm(confusion_data)):
        # Compute the confusion matrix
        cm = confusion_matrix(labels, predictions, labels=np.arange(num_author))

        # Replace zero counts with one to avoid division by zero
        cm = np.where(cm == 0, 1, cm)

        # Normalize the confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

        # Plot the heatmap without gridlines
        sns.heatmap(cm_normalized, cmap="Reds",
                    ax=axes[i], cbar=False,
                    xticklabels=False, yticklabels=False,  # Disable x and y tick labels
                    annot=False, linewidths=0, linecolor='gray', square=True)

        # Adjust font sizes and labels for clarity
        axes[i].tick_params(axis='both', which='major', length=0)  # Remove tick marks
        axes[i].set_title(name, fontsize=14)


    fig.tight_layout()
    plt.savefig('../analysis/blog50_vs_ccat50.pdf')


if __name__ == '__main__':

    # RESULT_1 = read_json('../output/imdb62/CIL/6_sessions/FT/results/5_session_result.json')
    # RESULT_2 = read_json('../output/imdb62/CIL/6_sessions/FT+/results/5_session_result.json')
    # RESULT_3 = read_json('../output/imdb62/CIL/6_sessions/FZ/results/5_session_result.json')
    # RESULT_4 = read_json('../output/imdb62/CIL/6_sessions/FZ+/results/5_session_result.json')
    # RESULT_5 = read_json('../output/imdb62/CIL/6_sessions/MAS/results/5_session_result.json')
    # RESULT_6 = read_json('../output/imdb62/CIL/6_sessions/EWC/results/5_session_result.json')
    # RESULT_7 = read_json('../output/imdb62/CIL/6_sessions/LWF/results/5_session_result.json')
    # RESULT_8 = read_json('../output/imdb62/CIL/6_sessions/LWF_E2/results/5_session_result.json')
    # RESULT_9 = read_json('../output/imdb62/CIL/6_sessions/FT_E2/results/5_session_result.json')
    #
    #
    # RESULT_10 = read_json('../output/imdb62/CIL/10_sessions/FT/results/9_session_result.json')
    # RESULT_11 = read_json('../output/imdb62/CIL/10_sessions/FT+/results/9_session_result.json')
    # RESULT_12 = read_json('../output/imdb62/CIL/10_sessions/FZ/results/9_session_result.json')
    # RESULT_13 = read_json('../output/imdb62/CIL/10_sessions/FZ+/results/9_session_result.json')
    # RESULT_14 = read_json('../output/imdb62/CIL/10_sessions/MAS/results/9_session_result.json')
    # RESULT_15 = read_json('../output/imdb62/CIL/10_sessions/EWC/results/9_session_result.json')
    # RESULT_16 = read_json('../output/imdb62/CIL/10_sessions/LWF/results/9_session_result.json')
    # RESULT_17 = read_json('../output/imdb62/CIL/10_sessions/LWF_E2/results/9_session_result.json')
    # RESULT_18 = read_json('../output/imdb62/CIL/10_sessions/FT_E2/results/9_session_result.json')

    # ITEMS = []
    #
    # for NAME, I in zip(['FT', 'FT+', 'FZ', 'FZ+', 'MAS', 'EWC', 'LWF', 'LWF_E2', 'FT_E2',
    #                     'FT', 'FT+', 'FZ', 'FZ+', 'MAS', 'EWC', 'LWF', 'LWF_E2', 'FT_E2'],
    #                    [RESULT_1, RESULT_2, RESULT_3, RESULT_4, RESULT_5, RESULT_6, RESULT_7, RESULT_8, RESULT_9,
    #                     RESULT_10, RESULT_11, RESULT_12, RESULT_13, RESULT_14, RESULT_15, RESULT_16, RESULT_17, RESULT_18]):
    #
    #     PREDICTIONS = I['pred_inc_ides']
    #     LABELS = I['true_inc_ides']
    #     NUM_AUTHOR = len(set(I['true_inc_ides']))
    #     ITEMS.append([PREDICTIONS, LABELS, NUM_AUTHOR, NAME])



    RESULT_1 = read_json('../output/blog50/CIL/6_sessions/FT+/results/5_session_result.json')
    RESULT_2 = read_json('../output/blog50/CIL/6_sessions/EWC/results/5_session_result.json')
    RESULT_3 = read_json('../output/blog50/CIL/6_sessions/MAS/results/5_session_result.json')
    RESULT_4 = read_json('../output/blog50/CIL/6_sessions/LWF/results/5_session_result.json')
    RESULT_5 = read_json('../output/blog50/CIL/6_sessions/FT_E2/results/5_session_result.json')

    RESULT_6 = read_json('../output/ccat50/CIL/6_sessions/FT+/results/5_session_result.json')
    RESULT_7 = read_json('../output/ccat50/CIL/6_sessions/EWC/results/5_session_result.json')
    RESULT_8 = read_json('../output/ccat50/CIL/6_sessions/MAS/results/5_session_result.json')
    RESULT_9 = read_json('../output/ccat50/CIL/6_sessions/LWF/results/5_session_result.json')
    RESULT_10 = read_json('../output/ccat50/CIL/6_sessions/FT_E2/results/5_session_result.json')


    ITEMS = []
    for NAME, I in zip(['FT+', 'EWC', 'MAS', 'LWF', 'FT_E2',
                        'FT+', 'EWC', 'MAS', 'LWF', 'FT_E2'],
                       [RESULT_1, RESULT_2, RESULT_3, RESULT_4, RESULT_5,
                        RESULT_6, RESULT_7, RESULT_8, RESULT_9, RESULT_10]):

        PREDICTIONS = I['pred_inc_ides']
        LABELS = I['true_inc_ides']
        NUM_AUTHOR = len(set(I['true_inc_ides']))
        ITEMS.append([PREDICTIONS, LABELS, NUM_AUTHOR, NAME])

    plot_heatmap(ITEMS)
