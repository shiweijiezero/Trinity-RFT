import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_trajectory_schematic():
    # 设置风格
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.linewidth': 1.5,
        'xtick.major.size': 0,
        'ytick.major.size': 0,
        'xtick.labelsize': 0,
        'ytick.labelsize': 0
    })

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-3, 5)

    # 去掉边框，只保留画布
    ax.axis('off')

    # 定义关键点坐标
    start_pos = (0, 0)
    pivot_pos = (4, 0.5)
    fail_pos = (9, -2)
    success_pos = (9, 3)
    reflect_pos = (9.5, -2.5) # 反思发生的位置

    # -------------------------------------------------------
    # 1. Base Trajectory (Exploration) - 橙色
    # -------------------------------------------------------
    # 路径：起点 -> Pivot -> 失败
    # 使用贝塞尔曲线风格或者简单的折线
    x_base = np.linspace(start_pos[0], fail_pos[0], 100)
    # 简单的二次曲线模拟轨迹
    y_base = -0.05 * x_base**2 + 0.2 * x_base

    ax.plot(x_base, y_base, color='#ff7f0e', linewidth=3, alpha=0.6, label='(a) Base Trajectory')
    ax.text(fail_pos[0], fail_pos[1]-0.3, 'Failure ✗', color='#ff7f0e', fontsize=14, fontweight='bold', ha='center')

    # -------------------------------------------------------
    # 2. Reflection (Diagnosis) - 紫色
    # -------------------------------------------------------
    # 在失败点旁边画一个框，表示生成的 Reflection 文本
    rect = patches.FancyBboxPatch((8.2, -3.8), 2.6, 1.2, boxstyle="round,pad=0.1",
                                  linewidth=1.5, edgecolor='#9467bd', facecolor='#f4efff')
    ax.add_patch(rect)
    ax.text(9.5, -3.2, '(b) Reflection\n"Error at Step t..."\n"Plan: Do X instead"',
            color='#5e3c99', fontsize=10, ha='center', va='center', style='italic')

    # 画一条虚线把失败点连接到反思，再连回 Pivot
    ax.annotate("", xy=pivot_pos, xytext=(8.2, -3.2),
                arrowprops=dict(arrowstyle="->", color='#9467bd', lw=1.5, ls='--'))

    ax.text(6, -1.5, 'Feedback / Guidance ($p_i$)', color='#9467bd', fontsize=10, rotation=20)

    # -------------------------------------------------------
    # 3. Retry with Guidance (Teacher) - 浅蓝色虚线背景
    # -------------------------------------------------------
    # 路径：Pivot -> 成功
    x_retry = np.linspace(pivot_pos[0], success_pos[0], 100)
    # 模拟生成的路径
    y_retry = 0.1 * (x_retry - 4)**2 + 0.5

    # 这条线代表 Teacher Distribution pi(y | x, plan)
    ax.plot(x_retry, y_retry, color='#aec7e8', linewidth=6, alpha=0.4)
    ax.text(7, 3.5, '(c) Teacher Path\n(Conditioned on Plan)', color='#6b8caf', fontsize=10, ha='center')

    # -------------------------------------------------------
    # 4. Distilled Retry (Student Target) - 深蓝色实线
    # -------------------------------------------------------
    # 在 Teacher 路径上画实线，代表这是我们想要 Student 学的
    ax.plot(x_retry, y_retry, color='#1f77b4', linewidth=2.5, label='(d) Distilled Target')
    ax.text(success_pos[0], success_pos[1]+0.3, 'Success ★', color='#1f77b4', fontsize=14, fontweight='bold', ha='center')

    # -------------------------------------------------------
    # 5. 关键节点标注 (Pivot)
    # -------------------------------------------------------
    # Start Node
    circle_start = patches.Circle(start_pos, 0.2, color='black')
    ax.add_patch(circle_start)
    ax.text(start_pos[0], start_pos[1]+0.3, 'Input ($x$)', ha='center', fontweight='bold')

    # Pivot Node
    circle_pivot = patches.Circle(pivot_pos, 0.2, color='#d62728', zorder=10)
    ax.add_patch(circle_pivot)
    ax.text(pivot_pos[0], pivot_pos[1]-0.5, 'Pivot Step ($t_{pivot}$)', color='#d62728', fontweight='bold', ha='center')

    # Shared Prefix Annotation
    ax.text(2, 0.8, 'Shared Prefix ($h_{<t}$)', color='gray', fontsize=10, ha='center')
    # 只有 Pivot 之后才分叉

    # -------------------------------------------------------
    # Context Distillation 示意
    # -------------------------------------------------------
    # 用一个括号或者标注说明：这里的训练 Loss
    # Loss = KL( Teacher || Student )

    # 在蓝色路径旁边加公式
    ax.text(5.5, 2.0, r"Optimization Target:", color='#1f77b4', fontsize=11, fontweight='bold')
    ax.text(5.5, 1.4, r"Input: $[x, h_{<t}]$ (No Plan!)", color='#333', fontsize=10)
    ax.text(5.5, 0.9, r"Label: Blue Path", color='#333', fontsize=10)

    # 图例
    # 自定义图例句柄
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='#ff7f0e', lw=3),
                    Line2D([0], [0], color='#9467bd', lw=2, ls='--'),
                    Line2D([0], [0], color='#aec7e8', lw=4),
                    Line2D([0], [0], color='#1f77b4', lw=3)]

    ax.legend(custom_lines,
              ['(a) Base Trajectory (Failed)',
               '(b) Reflection & Guidance',
               '(c) Teacher (with Plan)',
               '(d) Student Target (Distilled)'],
              loc='upper left', frameon=True, fontsize=11)

    plt.tight_layout()
    plt.savefig('trajectory_types.png', dpi=300, bbox_inches='tight')
    print("图像已生成: trajectory_types.png")
    plt.show()

if __name__ == "__main__":
    draw_trajectory_schematic()