import matplotlib.pyplot as plt
import numpy as np

ralm_data = {
    'Edge-focused': [40.6, 40.3, 40.5, 40.2, 40.35, 40.4, 40.25, 40.45],
    'Uniform': [42, 41.7, 42.3, 42.2, 41.5, 41.9, 42.5, 41.8, 42.1, 41.6, 42.4],
    'Middle-focused': [6.2, 18.9, 29.6, 37.4, 43.6, 49.8, 58.4, 66.3, 72.5, 76.2, 79.8]
}

# 更新后的标准差数据
std_devs = {
    'Edge-focused': [1.7, 1.63, 1.68, 1.65, 1.67, 1.66, 1.64, 1.69],
    'Uniform': [0.55, 0.5, 0.59, 0.55, 0.53, 0.57, 0.59, 0.52, 0.56, 0.51, 0.58],
    'Middle-focused': [4.45, 4.1, 3.89, 5.1, 5.5, 6.22, 4.34, 3.41, 3.58, 3.95, 3.75]
}

x_values = {
    'Edge-focused': [0, 1.43, 2.86, 4.29, 5.71, 7.14, 8.57, 10],
    'Uniform': [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30],
    'Middle-focused': [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
}

# 图表绘制代码保持不变
fig, ax = plt.subplots(figsize=(15, 9))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for (label, data), color in zip(ralm_data.items(), colors):
    x = x_values[label]
    y = data
    std = std_devs[label]

    ax.plot(x, y, marker='o', label=label, linewidth=3, markersize=10, color=color)
    ax.fill_between(x, np.array(y) - np.array(std), np.array(y) + np.array(std),
                    alpha=0.3, color=color)

ax.set_xlabel('Cumulative attention weight on relevant passages', fontsize=35)
ax.set_ylabel('RALM accuracy (%)', fontsize=38)
ax.legend(fontsize=34, loc='lower right')
ax.grid(True, linestyle='--', alpha=0.7)

ax.tick_params(axis='both', which='major', labelsize=38)

plt.tight_layout()
plt.savefig('ralm_accuracy_plot.pdf', bbox_inches='tight')