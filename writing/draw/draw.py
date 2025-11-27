import pandas as pd
import matplotlib.pyplot as plt
import os

# =================配置区域=================

# 1. 设置全局绘图风格 (字号进一步加大)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 16,                   # 基础字号加大
    'axes.titlesize': 22,              # 标题字号加大
    'axes.labelsize': 20,              # 轴标签加大
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'lines.linewidth': 3.5,            # 线条更粗
    'figure.titlesize': 26
})

# 2. 定义统一颜色
COLOR_R3L = '#1f77b4'      # 蓝色 (R3L)
COLOR_BASE = '#ff7f0e'     # 橙色 (GRPO)

# 3. 配置文件名与属性
FILES_CONFIG = {
    # ------------------- R3L 数据 (蓝色) -------------------
    'wandb_export_2025-11-20T11_56_06.479+08_00.csv': { # Reward
        'label': 'R3L',
        'group': 'Reward',
        'color': COLOR_R3L
    },
    'wandb_export_2025-11-20T11_55_46.588+08_00.csv': { # Ref KL
        'label': 'R3L',
        'group': 'Ref KL',
        'color': COLOR_R3L
    },
    'wandb_export_2025-11-20T11_55_50.693+08_00.csv': { # Grad Norm
        'label': 'R3L',
        'group': 'Grad Norm',
        'color': COLOR_R3L
    },
    'wandb_export_2025-11-20T11_52_34.277+08_00.csv': { # OPMD Loss (放入对比组)
        'label': 'R3L',
        'group': 'Loss Comparison',
        'color': COLOR_R3L
    },

    # ------------------- GRPO 数据 (橙色) -------------------
    'wandb_export_2025-11-26T14_57_08.972+08_00.csv': { # Reward
        'label': 'GRPO',
        'group': 'Reward',
        'color': COLOR_BASE
    },
    'wandb_export_2025-11-20T11_54_20.100+08_00.csv': { # Ref KL
        'label': 'GRPO',
        'group': 'Ref KL',
        'color': COLOR_BASE
    },
    'wandb_export_2025-11-20T11_54_47.188+08_00.csv': { # Grad Norm
        'label': 'GRPO',
        'group': 'Grad Norm',
        'color': COLOR_BASE
    },
    'wandb_export_2025-11-20T11_54_23.568+08_00.csv': { # PG Loss (放入对比组)
        'label': 'GRPO',
        'group': 'Loss Comparison',
        'color': COLOR_BASE
    },
    'wandb_export_2025-11-20T11_53_55.480+08_00.csv': { # PPO KL (Step KL)
        'label': 'GRPO Approx KL',
        'group': 'Update KL',
        'color': COLOR_BASE
    },
    'wandb_export_2025-11-20T11_54_30.871+08_00.csv': { # Clip Frac
        'label': 'GRPO Clip Fraction',
        'group': 'Clip Frac',
        'color': COLOR_BASE
    }
}

# =================逻辑区域=================

def load_data(filename):
    if not os.path.exists(filename):
        print(f"警告: 文件 {filename} 未找到，跳过。")
        return None
    try:
        df = pd.read_csv(filename)
        data = df.iloc[:, [0, 1]].copy()
        data.columns = ['Step', 'Value']
        return data
    except Exception as e:
        print(f"读取 {filename} 出错: {e}")
        return None

def plot_all():
    # 画布大小
    fig = plt.figure(figsize=(26, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.25)

    # 定义子图 (a)-(f)
    # 第一行
    ax_reward = fig.add_subplot(gs[0, 0])
    ax_ref_kl = fig.add_subplot(gs[0, 1])
    ax_grad = fig.add_subplot(gs[0, 2])
    # 第二行
    ax_loss_comp = fig.add_subplot(gs[1, 0])  # 合并的 Loss 图
    ax_update_kl = fig.add_subplot(gs[1, 1])  # PPO Update KL
    ax_clip = fig.add_subplot(gs[1, 2])       # Clip Fraction

    # 映射配置
    axes_map = {
        'Reward': ax_reward,
        'Ref KL': ax_ref_kl,
        'Grad Norm': ax_grad,
        'Loss Comparison': ax_loss_comp,
        'Update KL': ax_update_kl,
        'Clip Frac': ax_clip
    }

    # 设置标题 (严格区分两个KL)
    ax_reward.set_title("(a) Average Reward")
    ax_ref_kl.set_title(r"(b) Reference KL ($\pi_\theta$ vs $\pi_{ref}$)")
    ax_grad.set_title("(c) Gradient Norm")
    ax_loss_comp.set_title("(d) Policy Loss Comparison")
    ax_update_kl.set_title(r"(e) GRPO Update KL ($\pi_{new}$ vs $\pi_{old}$)")
    ax_clip.set_title("(f) GRPO Clip Fraction")

    for fname, config in FILES_CONFIG.items():
        df = load_data(fname)
        if df is None: continue

        group = config['group']
        label = config['label']
        color = config['color']

        # 核心逻辑：如果是橙色(GRPO)，增加透明度
        alpha_val = 0.6 if color == COLOR_BASE else 1.0

        if group in axes_map:
            ax = axes_map[group]
            ax.plot(df['Step'], df['Value'], label=label, color=color, alpha=alpha_val)

    # 统一设置图例、网格和标签
    all_axes = [ax_reward, ax_ref_kl, ax_grad, ax_loss_comp, ax_update_kl, ax_clip]

    for ax in all_axes:
        ax.set_xlabel("Training Steps")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='best', frameon=True, fancybox=True)
        # 显式设置所有边框可见
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

    plt.tight_layout()
    output_file = 'training_analysis_final.png'
    plt.savefig(output_file, dpi=500, bbox_inches='tight')
    print(f"绘图完成，已保存为 {output_file}")
    plt.show()

if __name__ == "__main__":
    plot_all()