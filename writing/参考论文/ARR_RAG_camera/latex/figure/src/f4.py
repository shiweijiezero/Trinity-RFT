import matplotlib.pyplot as plt

pie_data = [87.37, 5.97, 6.67]
pie_labels = ['Edge-focused', 'Uniform', 'Middle-focused']

# 使用更浅的颜色
colors = {
    'start': '#FF8C98',  # 浅蓝色
    'middle': '#7FD4FF',  # 浅橙色
    'end': '#86E088'  # 浅绿色
}

fig, ax = plt.subplots(figsize=(16, 12))  # 保持方形图形大小
pie_colors = [colors['start'], colors['middle'], colors['end']]
wedges, texts, autotexts = ax.pie(pie_data, labels=pie_labels, autopct='%1.1f%%', startangle=90, colors=pie_colors, textprops={'fontsize': 44})

# 将图例移到顶部
ax.legend(wedges, pie_labels,
          loc="upper center",
          bbox_to_anchor=(0.1, 1.1),
          ncol=1,  # 将图例项目排列在一行
          fontsize=43)

plt.setp(autotexts, size=44)

plt.tight_layout()
plt.savefig('attention_distribution_pie.pdf', bbox_inches='tight')