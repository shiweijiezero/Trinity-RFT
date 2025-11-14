import matplotlib.pyplot as plt
import numpy as np

edge_focused = [26, 22, 17, 1, 1.5, 0.7, 1.5, 0.4, 0.8, 8, 12, 16]

colors = {'start': '#FF8C98', 'middle': '#7FD4FF', 'end': '#86E088'}

def create_bar_plot(data, title):
    fig, ax = plt.subplots(figsize=(12, 10))  # Further increased figure size
    total = sum(data)
    percentages = data

    # Adjusted bar width
    bar_width = 0.8

    ax.bar(range(3), percentages[:3], color=colors['start'], label='Start Tokens', width=bar_width)
    ax.bar(range(3, 9), percentages[3:9], color=colors['middle'], label='Middle Tokens', width=bar_width)
    ax.bar(range(9, 12), percentages[9:], color=colors['end'], label='End Tokens', width=bar_width)

    ax.set_xlabel('Input Context', fontsize=50)  # Increased font size
    ax.set_ylabel('Attention Weight (%)', fontsize=50)  # Increased font size
    ax.set_ylim(0, max(percentages) * 1.2)

    ax.set_xticks([1, 6, 10])
    ax.set_xticklabels(['Start', 'Middle', 'End'], fontsize=44)  # Increased font size

    # Increase y-axis tick label size
    ax.tick_params(axis='y', labelsize=44)  # Increased font size

    start = 65.2
    middle = 4.1
    end = 30.7

    legend_labels = [
        f'Start Tokens ({start:.1f}%)',
        f'Middle Tokens ({middle:.1f}%)',
        f'End Tokens ({end:.1f}%)'
    ]
    ax.legend(labels=legend_labels, loc='upper right', fontsize=43)  # Increased font size

    plt.tight_layout()
    plt.savefig('edge_focused_attention.pdf', bbox_inches='tight')

create_bar_plot(edge_focused, 'Edge-focused Attention')