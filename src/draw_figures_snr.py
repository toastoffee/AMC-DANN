import torch
from torch import nn, optim
import warnings

import matplotlib.pyplot as plt
import itertools
from train.device_utils import get_device
from dataset.dataloader_helper import DataloaderHelper
from train.train_classics import eval_and_get_acc

from module.dann import DANN, DANN_wrapper
from module.dan import ImageClassifier, DAN_wrapper
from module.adda import ADDA
from module.mcc import MCC
from module.mcd import MCD, MCD_wrapper
from module.sdidn import DistanDANN, SDIDN_wrapper

from module.loss import CovarianceOrthogonalLoss, DomainContrastiveLoss
from module.grl import GradientReversalFunction
from module.utils import freeze, unfreeze
from dataset.dataset_utils import set_seeds
import pickle

import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import rcParams

# 设置全局字体为系统中存在的字体，比如 Liberation Serif 或 Nimbus Roman No9 L
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 12  # 可根据需要调整大小


def draw_snrs_all(*accuracies, labels=None, save_path='./figures/RML2016c.pdf'):  # 修改为保存为 PDF
    """
    绘制多模型 SNR-准确率曲线，模仿 IEEE 论文风格（虚线 + 小标记）。

    Args:
        *accuracies: 每个是 dict{snr: acc}，snr 为 int，acc 为 float ∈ [0,1]
        labels (list): 模型名称列表
        save_path (str): 保存路径，默认为 PDF 格式
    """
    if not accuracies:
        raise ValueError("At least one accuracy dict must be provided.")

    num_models = len(accuracies)
    if labels is None:
        labels = [f'Model_{i}' for i in range(num_models)]
    elif len(labels) != num_models:
        raise ValueError("Number of labels must match number of accuracy dicts.")

    snr_range = list(range(-20, 22, 2))

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = [
        'darkgreen',  # 示例颜色
        'darkred',
        'darkorange',
        'darkblue',
        'darkviolet',
        'crimson',
        'gray',
        'purple',
        'teal',
        'royalblue'
    ]

    markers = ['^', 'o', 's', 'x', '+', 'D', '*', '<', '>', 'p']

    for i, acc_dict in enumerate(accuracies):
        x = []
        y = []
        for snr in sorted(acc_dict.keys()):
            if snr in snr_range:
                x.append(snr)
                y.append(acc_dict[snr])

        ax.plot(x, y,
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                linewidth=1.5,
                markersize=3,
                linestyle='--',
                label=labels[i],
                alpha=0.9)

    ax.set_xlabel('SNR(dB)')
    ax.set_ylabel('Accuracy')
    ax.set_xlim(-20, 22)
    ax.set_ylim(0, 0.5)
    ax.grid(True, which='both', linestyle='--', alpha=0.6, linewidth=0.5, color='gray')

    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=False)

    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')  # 保存为矢量 PDF 文件
    plt.close()


def plot_from_saved(save_file: str = "./results/accuracies.pkl",
                    figure_path: str = "./figures/RML2022_accuracy_vs_snr.pdf"):
    """
    从保存的 .pkl 文件中读取准确率，并绘图。
    """
    with open(save_file, 'rb') as f:
        results = pickle.load(f)

    # 按固定顺序提取（避免字典乱序）
    model_order = ['SDIDN', 'ADDA', 'DANN', 'DAN', 'MCC', 'MCD']
    accuracies = [results[name] for name in model_order if name in results]
    labels = [name for name in model_order if name in results]

    # 调用你的绘图函数
    draw_snrs_all(*accuracies, labels=labels, save_path=figure_path)
    print(f"✅ Figure saved to {figure_path}")


if __name__ == "__main__":
    plot_from_saved("./results/16a_22_accs.pkl", "./figures/RML2022_accuracy_vs_snr.pdf")
