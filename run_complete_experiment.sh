#!/bin/bash

# 设置随机种子
export PYTHONHASHSEED=42
export CUDA_LAUNCH_BLOCKING=1

echo "开始完整的 Transformer 实验..."

# 创建目录
python create_dirs.py

# 训练完整模型
echo "1. 训练完整模型..."
python scripts/train.py \
    --config local \
    --local-data "data/custom_dataset" \
    --data-format json \
    --train-size 49000 \
    --val-size 1000 \
    --epochs 10 \
    --batch-size 32

# 运行消融实验
echo "2. 运行消融实验..."
python scripts/ablation_study.py

# 综合评估
echo "3. 综合评估模型性能..."
python scripts/evaluate_metrics.py

# 生成图表
echo "4. 生成实验图表..."
python scripts/plot_training_curves.py

echo "实验完成！所有结果保存在 results/ 目录中"