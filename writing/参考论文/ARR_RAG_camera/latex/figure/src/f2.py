import matplotlib.pyplot as plt
import numpy as np

uniform = [0.1, 0.07, 0.03, 11.2, 7.4, 6.6, 7.8, 9.4, 8.3, 0.1, 0.2, 0.25]

colors = {'start': '#FF8C98', 'middle': '#7FD4FF', 'end': '#86E088'}

def create_bar_plot(data, title):
    fig, ax = plt.subplots(figsize=(12, 10))  # Increased figure size
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

    start = 0.1
    middle = 99.8
    end = 0.1

    legend_labels = [
        f'Start Tokens ({start:.1f}%)',
        f'Middle Tokens ({middle:.1f}%)',
        f'End Tokens ({end:.1f}%)'
    ]

    # 添加透明度设置
    legend = ax.legend(labels=legend_labels, loc='upper right', fontsize=43)
    legend.get_frame().set_alpha(0.4)  # 设置图例背景的透明度

    plt.tight_layout()
    plt.savefig('uniform_attention.pdf', bbox_inches='tight')

create_bar_plot(uniform, 'Uniform Attention')