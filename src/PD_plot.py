import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import read_json
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines


def plot_horizontal_histogram(items):
    # Define the specific dark colors
    primary_colors = [
        '#1E90FF',  # DarkBlue
        '#FA8072',  # DarkRed
        '#32CD32',  # DarkGreen
        '#FF8C00',  # DarkOrange
        '#8A2BE2',  # DarkPurple
        '#FFD700',  # DarkYellow
        '#66CDAA',  # DarkGray
        '#FF4500'  # OrangeRed
    ]

    # Get the order of unique keys based on their first appearance in items
    unique_keys = []
    for values in items.values():
        for key in values.keys():
            if key not in unique_keys:
                unique_keys.append(key)

    num_keys = len(unique_keys)
    colors = primary_colors[:num_keys]

    # Prepare data for plotting
    bar_height = 0.8  # Adjust height of the bars
    y_positions = []
    color_map = {key: colors[i] for i, key in enumerate(unique_keys)}

    # Plot each dataset with consistent item colors in the correct order
    for i, (name, values) in enumerate(items.items()):
        y_pos = np.arange(len(values)) + i * (num_keys + 1)
        # Ensure the bars are plotted in the same order as unique_keys, but reverse the order for plotting
        values_list = [values.get(key, 0) for key in unique_keys[::-1]]  # Reverse order of bars
        colors_for_bars = [color_map[key] for key in unique_keys[::-1]]  # Reverse order of colors
        plt.barh(y_pos, values_list, color=colors_for_bars, edgecolor='black', height=bar_height)

        # Store the y-positions for the dataset labels
        y_positions.append(np.mean(y_pos))

    # Create legend handles with circles in the order of appearance
    legend_handles = [
        mlines.Line2D([], [], color=color_map[key], marker='o', linestyle='None', markersize=10, label=key)
        for key in unique_keys
    ]

    # Adding legend, labels, and title
    plt.legend(handles=legend_handles, loc='upper right', framealpha=0.5,
               bbox_to_anchor=(1.01, 1))  # Adjust legend position slightly lower
    plt.yticks(y_positions, items.keys())  # Set y-ticks at the center positions of each dataset
    plt.tight_layout()  # Adjust layout to fit everything within the figure

    # Save the figure without margins
    plt.savefig('../analysis/PD_plot.pdf', bbox_inches='tight', pad_inches=0)

# plt.savefig('../analysis/PD_plot.pdf')

if __name__ == '__main__':

    EXEMPLARS_MODELS = ['FT_E2', 'FT_E5', 'FT_E10', 'FT_E20', 'LWF_E2', 'LWF_E5', 'LWF_E10', 'LWF_E20']
    DATASETS = ['blog50', 'imdb62']

    ITEMS = {}
    for d in DATASETS:

        ITEMS[d] = {}
        BASE_ACC = read_json(f'../output/{d}/CIL/6_sessions/FT/results/0_session_result.json')['accuracy']

        for m in EXEMPLARS_MODELS:
            CURR_ACC = read_json(f'../output/{d}/CIL/6_sessions/{m}/results/5_session_result.json')['accuracy']
            ACC = BASE_ACC - CURR_ACC

            ITEMS[d].update({m: ACC})

    plot_horizontal_histogram(ITEMS)
