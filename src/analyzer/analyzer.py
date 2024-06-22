import os
import pandas as pd
from itertools import accumulate
import matplotlib.pyplot as plt
from src.utils import read_json


def plot_histogram(data1, data2, filename='../analysis/histogram.pdf'):
    def prepare_data(data):
        df = pd.DataFrame()
        author_order = []

        for session in data:
            for author_data in session:
                author_name = list(author_data.keys())[0]
                if author_name not in author_order:
                    author_order.append(author_name)

        for i, session in enumerate(data, 1):
            session_data = {list(author_data.keys())[0]: list(author_data.values())[0] for author_data in session}
            document_counts = [session_data.get(author, 0) for author in author_order]
            df[f'session_{i}'] = pd.Series(document_counts, index=author_order)

        return df, author_order

    df1, author_order1 = prepare_data(list(data1.values())[0])
    df2, author_order2 = prepare_data(list(data2.values())[0])

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))  # Reduced size

    def plot_data(ax, df, data_name, show_y_axis=True):
        dark_colors = ['#00389F', '#59006E', '#3D5F74', '#006da4', '#2E4053', '#CBC3E3']
        for i, session_color in enumerate(dark_colors[:len(df.columns)]):
            ax.bar(range(len(df)), df[f'session_{i + 1}'], color=session_color, width=0.7, align='center',
                   # Reduced width
                   label=f'Session {i}')

        ax.set_xlabel('Number of Authors', fontsize=10)  # Adjusted font size
        if show_y_axis:
            ax.set_ylabel('Number of Documents', fontsize=10)  # Adjusted font size
        ax.set_title(data_name, fontsize=12)  # Adjusted font size

        xticks_labels = [f"{i + 1}" for i, name in enumerate(df.index)]
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(xticks_labels, rotation=0, ha='center', fontsize=5)  # Adjusted font size
        ax.tick_params(axis='x', pad=5)

        session_changes = [i for i in range(len(df.columns) - 1) if df.columns[i][:7] != df.columns[i + 1][:7]]
        for change in session_changes:
            ax.axvline(x=change, color='red', linestyle='--', linewidth=0.5)
            ax.text(change + 0.5, -100, f'Session {change + 2}', ha='center', fontsize=8, color='darkblue')

        ax.set_ylim(0, 3500)
        ax.margins(x=0.01)  # Adjusted margins

    plot_data(axs[0], df1, list(data1.keys())[0])
    plot_data(axs[1], df2, list(data2.keys())[0], show_y_axis=False)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), fontsize=10)  # Adjusted font size

    # Hide tick and tick label of the big axis
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename, bbox_inches='tight')
    plt.show()


def plot_incremental_learning(authors1, accuracies1, dataset1, authors2, accuracies2, dataset2,
                              file_path='../analysis/inc_acc.pdf'):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # First subplot
    for label, acc in accuracies1.items():
        axs[0].plot(authors1, list(acc.values()), marker='o', linestyle='-', label=label)
    axs[0].set_title(dataset1, fontsize=14)
    axs[0].set_xlabel('Number of Authors', fontsize=14)
    axs[0].set_ylabel('Accuracy (%)', fontsize=14)
    axs[0].grid(True)
    axs[0].set_xticks(authors1)
    for author in authors1:
        axs[0].axvline(x=author, color='gray', linestyle='--', linewidth=0.6)

    # Second subplot
    for label, acc in accuracies2.items():
        axs[1].plot(authors2, list(acc.values()), marker='o', linestyle='-', label=label)
    axs[1].set_title(dataset2, fontsize=14)
    axs[1].set_xlabel('Number of Authors', fontsize=14)
    axs[1].set_ylabel('Accuracy (%)', fontsize=14)
    axs[1].grid(True)
    axs[1].set_xticks(authors2)
    for author in authors2:
        axs[1].axvline(x=author, color='gray', linestyle='--', linewidth=0.6)

    # Combine legends, removing duplicates
    handles, labels = [], []
    for ax in axs:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.92])  # Adjust layout to make room for the legend
    plt.savefig(file_path)
    plt.show()


def extract_acc(dir_path, dataset):
    all_acc = {}
    num_authors_per_session = read_json(f'../data/CIL/{dataset}_CIL/authors_partition_config.json')
    num_authors_inc = list(accumulate(list(num_authors_per_session.values())))

    for root, dirs, _ in os.walk(dir_path):
        for d in dirs:
            result_dir = f'{root}/{d}/results'
            for _, _, files in os.walk(result_dir):
                for file in files:
                    data = read_json(os.path.join(result_dir, file))
                    model_name = d
                    session = data['session']  # Remove '.json' extension
                    if model_name not in all_acc:
                        all_acc[model_name] = {session: data['accuracy']}
                    else:
                        all_acc[model_name].update({session: data['accuracy']})

    return all_acc, num_authors_inc
