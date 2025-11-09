import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
from config.training_config import TrainingConfig, SmallConfig, LocalDatasetConfig, ChineseConfig
from data.tokenizer import build_tokenizer, TokenizerWrapper
# 关键修改1：导入 load_raw_dataset（加载原始数据集给分词器训练）
from data.dataset_loader import get_data_loaders, load_raw_dataset
from model.light_transformer import LightTransformer
from training.optimizer import get_optimizer_and_scheduler
from training.trainer import TransformerTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="训练Transformer模型")
    parser.add_argument("--config", type=str, default="default",
                        choices=["default", "small", "local", "chinese"],
                        help="配置类型")
    parser.add_argument("--dataset", type=str, default="iwslt2017",
                        help="数据集名称（在线数据集时使用）")
    parser.add_argument("--local-data", type=str,
                        help="本地数据集路径")
    parser.add_argument("--data-format", type=str, default="json",
                        choices=["json", "csv", "text"],
                        help="本地数据集格式")
    parser.add_argument("--train-file", type=str, default="train.json",
                        help="本地训练数据文件名")
    parser.add_argument("--val-file", type=str, default="validation.json",
                        help="本地验证数据文件名")
    parser.add_argument("--train-size", type=int, default=45000,
                        help="训练集大小")
    parser.add_argument("--val-size", type=int, default=5000,
                        help="验证集大小")
    parser.add_argument("--epochs", type=int, default=15,
                        help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="批次大小")
    return parser.parse_args()


def main():
    args = parse_args()

    # 选择配置
    if args.config == "small":
        config = SmallConfig()
    elif args.config == "local":
        config = LocalDatasetConfig()
        if args.local_data:
            config.LOCAL_DATA_PATH = args.local_data
        config.LOCAL_DATA_FORMAT = args.data_format
        config.LOCAL_TRAIN_FILE = args.train_file
        config.LOCAL_VAL_FILE = args.val_file
    elif args.config == "chinese":
        config = ChineseConfig()
    else:
        config = TrainingConfig()

    # 覆盖配置参数
    if not config.USE_LOCAL_DATA:
        config.DATASET_NAME = args.dataset

    config.TRAIN_SIZE = args.train_size
    config.VAL_SIZE = args.val_size
    config.NUM_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size

    # 打印配置信息
    print("=" * 60)
    print("训练配置信息")
    print("=" * 60)
    print(f"配置类型: {args.config}")
    if config.USE_LOCAL_DATA:
        print(f"数据集: 本地数据集")
        print(f"数据路径: {config.LOCAL_DATA_PATH}")
        print(f"数据格式: {config.LOCAL_DATA_FORMAT}")
        print(f"训练文件: {config.LOCAL_TRAIN_FILE}")
        print(f"验证文件: {config.LOCAL_VAL_FILE}")
    else:
        print(f"数据集: {config.DATASET_NAME}")
        print(f"数据集配置: {config.DATASET_CONFIG}")
    print(f"训练大小: {config.TRAIN_SIZE}")
    print(f"验证大小: {config.VAL_SIZE}")
    print(f"批次大小: {config.BATCH_SIZE}")
    print(f"训练轮数: {config.NUM_EPOCHS}")
    print(f"设备: {config.DEVICE}")
    print(f"混合精度: {config.MIXED_PRECISION}")
    print(f"词汇表配置大小: {config.VOCAB_SIZE}")
    print(f"分词器保存路径: {config.TOKENIZER_PATH}")
    print("=" * 60)

    # 构建分词器（关键修改2：加载原始数据集，传入分词器训练）
    print("构建分词器...")
    # 1. 加载原始数据集（未编码的文本，用于训练分词器）
    raw_dataset = load_raw_dataset(config)
    # 2. 传入 dataset 参数，训练分词器（解决词汇表过小问题）
    raw_tokenizer = build_tokenizer(config, dataset=raw_dataset)
    tokenizer = TokenizerWrapper(raw_tokenizer)

    # 更新词汇表大小（同步到 config，确保模型匹配）
    actual_vocab_size = tokenizer.get_vocab_size()
    if actual_vocab_size != config.VOCAB_SIZE:
        print(f"更新词汇表大小: {config.VOCAB_SIZE} -> {actual_vocab_size}")
        config.VOCAB_SIZE = actual_vocab_size

    # 加载数据（编码后的数据集，用于模型训练）
    print("加载数据集...")
    train_loader, val_loader = get_data_loaders(config, tokenizer)

    # 初始化模型（使用更新后的词汇表大小）
    print("初始化模型...")
    model = LightTransformer(config).to(config.DEVICE)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"模型词汇表大小: {config.VOCAB_SIZE}（与分词器一致）")

    # 初始化优化器和调度器
    print("初始化优化器...")
    optimizer, scheduler = get_optimizer_and_scheduler(model, config)

    # 初始化训练器
    trainer = TransformerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config
    )

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()