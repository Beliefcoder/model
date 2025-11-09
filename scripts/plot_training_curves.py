#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import matplotlib
# 远程服务器无图形界面，必须添加这行
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_training_curves():
    """Plot training curves (英文注释替换中文)"""
    
    # 模拟训练数据（不变）
    epochs = list(range(1, 11))
    train_losses = [4.2, 2.8, 2.1, 1.7, 1.5, 1.4, 1.3, 1.2, 1.2, 1.1]
    val_perplexities = [85.2, 42.3, 25.6, 18.9, 15.8, 14.7, 14.3, 14.2, 14.3, 14.4]
    learning_rates = [3e-5, 8e-5, 1.5e-4, 2.2e-4, 2.8e-4, 3e-4, 3e-4, 3e-4, 3e-4, 3e-4]
    
    # 创建图表（不变）
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # 训练损失（中文→英文）
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Train Loss')
    ax1.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 验证困惑度（中文→英文）
    ax2.plot(epochs, val_perplexities, 'r-', linewidth=2, label='Validation Perplexity')
    ax2.set_title('Validation Perplexity Curve', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Perplexity')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 学习率（中文→英文）
    ax3.plot(epochs, learning_rates, 'g-', linewidth=2, label='Learning Rate')
    ax3.set_title('Learning Rate Scheduling Curve', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Learning Rate')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    
    # 保存图表（不变）
    os.makedirs("results", exist_ok=True)
    plt.savefig('results/training_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/training_curves.pdf', bbox_inches='tight')
    plt.close()
    
    # 打印信息（中文→英文）
    print("Training curves saved to results/training_curves.png")

if __name__ == "__main__":
    plot_training_curves()