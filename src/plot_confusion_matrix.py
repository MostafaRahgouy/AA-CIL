import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import read_json
from sklearn.metrics import confusion_matrix


def plot_heatmap(confusion_data, num_classes):
    num_plots = len(confusion_data)
    num_cols = 4
    num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate the number of rows needed
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 6 * num_rows))

    # Flatten axes array if there is more than one row
    if num_rows > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, (predictions, labels, name) in enumerate(confusion_data):
        # Compute the confusion matrix
        cm = confusion_matrix(labels, predictions, labels=np.arange(num_classes))

        # Replace zero counts with one to avoid division by zero
        cm = np.where(cm == 0, 1, cm)

        # Normalize the confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

        # Plot the heatmap without gridlines
        sns.heatmap(cm_normalized, cmap="Reds",
                    ax=axes[i], cbar=False,
                    xticklabels=np.arange(1, num_classes + 1), yticklabels=np.arange(1, num_classes + 1),
                    annot=False, linewidths=0, linecolor='gray', square=True)

        # Adjust font sizes and labels for clarity
        axes[i].tick_params(axis='both', which='major', labelsize=6)
        axes[i].set_xlabel('Predicted Label', fontsize=9)
        axes[i].set_ylabel('True Label', fontsize=9)
        axes[i].set_title(name, fontsize=14)

    # Turn off any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    plt.savefig('../analysis/blog_con_result.pdf')


if __name__ == '__main__':
    RESULT_1 = read_json('../output/blog50/CIL/FT/results/5_session_result.json')
    RESULT_2 = read_json('../output/blog50/CIL/FZ/results/5_session_result.json')
    RESULT_3 = read_json('../output/blog50/CIL/FT+/results/5_session_result.json')
    RESULT_4 = read_json('../output/blog50/CIL/EWC/results/5_session_result.json')
    RESULT_5 = read_json('../output/blog50/CIL/LWF_E2/results/5_session_result.json')
    RESULT_6 = read_json('../output/blog50/CIL/LWF/results/5_session_result.json')
    RESULT_7 = read_json('../output/blog50/CIL/LWF_E10/results/5_session_result.json')
    RESULT_8 = read_json('../output/blog50/CIL/MAS/results/5_session_result.json')

    ITEMS = []
    # for IDX, I in enumerate([RESULT_1, RESULT_2, RESULT_3, RESULT_4, RESULT_5, RESULT_6]):

    for NAME, I in zip(['FT', 'FZ', 'FT+', 'EWC', 'LWF_E2', 'LWF', 'LWF_E10', 'MAS'],
                       [RESULT_1, RESULT_2, RESULT_3, RESULT_4, RESULT_5, RESULT_6, RESULT_7, RESULT_8]):
        PREDICTIONS = I['pred_inc_ides']
        LABELS = I['true_inc_ides']
        ITEMS.append([PREDICTIONS, LABELS, NAME])

    plot_heatmap(ITEMS, num_classes=50)
