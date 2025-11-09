 # Encoder-Decoder Transformer

## 项目概述
在NVIDIA RTX 4090 显卡上，2小时内完成Encoder-Decoder Transformer模型的完整训练。

## 核心特性
- 🚀 **高效训练**: 1.5小时内完成50个epoch训练
- 💾 **显存优化**: 混合精度训练 + 梯度检查点
- 📊 **轻量模型**: 5M参数，适配小数据集
- 📈 **稳定收敛**: AdamW + 学习率热身 + 梯度裁剪
- 🔍 **高级推理**: 支持Beam Search解码
- ⚡ **可选优化**: 支持Flash Attention（长序列）

## 快速开始

### 环境安装
```bash
pip install -r requirements.txt
