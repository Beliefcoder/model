class TrainingConfig:
    # 数据集配置（Hugging Face公开数据集）
    DATASET_NAME = "iwslt2017"  # 数据集名称
    DATASET_CONFIG = "iwslt2017-en-de"  # 英德翻译配置
    SRC_LANG = "en"  # 源语言（英语）
    TGT_LANG = "de"  # 目标语言（德语）

    # 本地数据集配置
    USE_LOCAL_DATA = False  # 是否使用本地数据集（默认：否）
    LOCAL_DATA_PATH = "data/custom_dataset"  # 本地数据集路径
    LOCAL_DATA_FORMAT = "json"  # 本地数据格式（json/csv/text）
    LOCAL_TRAIN_FILE = "train.json"  # 本地训练文件名
    LOCAL_VAL_FILE = "validation.json"  # 本地验证文件名

    # 数据规模配置
    TRAIN_SIZE = 45000  # 训练集样本数（截取部分加速训练）
    VAL_SIZE = 5000  # 验证集样本数
    MAX_LENGTH = 64  # 句子最大长度（超过截断，不足填充）
    VOCAB_SIZE = 8192  # BPE分词器词汇表大小（英德翻译推荐）

    # 模型结构配置
    D_MODEL = 256  # 模型 embedding 维度
    NHEAD = 4  # 多头注意力头数
    NUM_ENCODER_LAYERS = 2  # 编码器层数
    NUM_DECODER_LAYERS = 2  # 解码器层数
    DIM_FEEDFORWARD = 1024  # 前馈网络隐藏层维度
    DROPOUT = 0.1  # dropout 概率
    USE_FLASH_ATTENTION = False  # 是否使用FlashAttention（需GPU支持）

    # 训练超参数配置
    BATCH_SIZE = 32  # 批次大小
    NUM_EPOCHS = 15  # 训练轮数
    LEARNING_RATE = 3e-4  # 初始学习率
    WEIGHT_DECAY = 1e-4  # 权重衰减（正则化）
    GRADIENT_CLIP = 1.0  # 梯度裁剪阈值（防止梯度爆炸）
    NUM_WARMUP_STEPS = 100  # 学习率热身步数

    # 硬件与路径配置
    DEVICE = "cuda:0"  # 训练设备（cuda:0/CPU）
    NUM_WORKERS = 4  # 数据加载线程数
    MIXED_PRECISION = True  # 是否使用混合精度训练（加速+省显存）
    CHECKPOINT_PATH = "outputs/checkpoints/latest_transformer.pth"  # 模型权重保存路径
    LOG_DIR = "outputs/logs"  # 训练日志保存路径
    TOKENIZER_PATH = "outputs/tokenizer.json"  # 分词器保存路径


class SmallConfig(TrainingConfig):
    """小规模配置（快速测试用）"""
    TRAIN_SIZE = 1000  # 少量训练样本
    VAL_SIZE = 100  # 少量验证样本
    NUM_EPOCHS = 3  # 少量训练轮数
    BATCH_SIZE = 16  # 减小批次大小
    VOCAB_SIZE = 2048  # 减小词汇表（测试用）


class LocalDatasetConfig(TrainingConfig):
    """本地数据集专用配置（适配英德翻译格式）"""
    USE_LOCAL_DATA = True  # 启用本地数据
    LOCAL_DATA_PATH = "data/custom_dataset"  # 你的本地数据集文件夹路径
    LOCAL_DATA_FORMAT = "json"  # 固定为 json
    LOCAL_TRAIN_FILE = "train.json"  # 你的训练文件名（无需修改）
    LOCAL_VAL_FILE = "validation.json"  # 你的验证文件名（无需修改）
    TRAIN_SIZE = 30000  # 按你的数据集大小调整（比如有5万条就设为50000）
    VAL_SIZE = 3000  # 验证集大小
    SRC_LANG = "en"  # 源语言：英文（对应 dataset.translation.en）
    TGT_LANG = "de"  # 目标语言：德文（对应 dataset.translation.de）
    VOCAB_SIZE = 8192  # 英德翻译推荐词汇表大小
    MAX_LENGTH = 64  # 英文/德文句子长度适中，64足够


class ChineseConfig(TrainingConfig):
    """中英翻译专用配置（适配中文特性）"""
    # 数据集配置（opus100 英中平行语料）
    DATASET_NAME = "opus100"
    DATASET_CONFIG = "en-zh"
    SRC_LANG = "en"  # 源语言（英语）
    TGT_LANG = "zh"  # 目标语言（中文）

    # 中文适配配置
    VOCAB_SIZE = 10000  # 中文词汇表更大（字符+词语）
    MAX_LENGTH = 80  # 中文句子更长，放宽长度限制
    BATCH_SIZE = 24  # 中文编码后序列略长，减小批次避免OOM