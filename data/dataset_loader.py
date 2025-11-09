import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset
import warnings

# 忽略无关警告
warnings.filterwarnings("ignore")


def load_raw_dataset(config):
    """加载原始平行语料（未编码，用于分词器训练）
    适配两种数据源：
    1. 本地数据集（{"translation": {"en": "", "de": ""}} 格式）
    2. Hugging Face 公开数据集
    """
    if config.USE_LOCAL_DATA:
        return _load_local_raw_dataset(config)
    else:
        return _load_hf_raw_dataset(config)


def _load_local_raw_dataset(config, is_val=False):
    """加载本地原始数据集（核心适配：translation.en/de 嵌套格式）"""
    # 选择训练/验证文件
    filename = config.LOCAL_VAL_FILE if is_val else config.LOCAL_TRAIN_FILE
    data_path = os.path.join(config.LOCAL_DATA_PATH, filename)
    
    # 校验文件是否存在
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ 本地文件不存在：{data_path}")

    raw_dataset = []  # 存储原始解析数据
    total_lines = 0  # 原始总行数
    parse_error_count = 0  # JSON解析错误数
    format_error_count = 0  # 字段格式错误数

    print(f"\n📥 加载本地数据集：{data_path}")
    print(f"   适配格式：{{'translation': {{'en': '英文', 'de': '德文'}}}}")

    with open(data_path, "r", encoding="utf-8") as f:
        file_content = f.read().strip()
        data_list = []

        # 情况1：文件是完整的JSON数组（被 [] 包裹）
        if file_content.startswith("[") and file_content.endswith("]"):
            try:
                data_list = json.loads(file_content)
                total_lines = len(data_list)
                print(f"   检测到JSON数组格式，共 {total_lines} 条数据")
            except json.JSONDecodeError as e:
                raise RuntimeError(f"❌ JSON数组格式错误：{str(e)}（请检查括号是否匹配、逗号是否多余）") from e
        
        # 情况2：文件是每行一个JSON对象（含可能的逗号分隔符）
        else:
            lines = file_content.split("\n")
            total_lines = len(lines)
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                # 跳过空行、数组元素间的逗号、注释行
                if not line or line.startswith(",") or line.startswith("//"):
                    continue
                try:
                    data = json.loads(line)
                    data_list.append(data)
                except json.JSONDecodeError:
                    parse_error_count += 1
                    continue

    # 提取有效数据（适配 translation.en/de 格式，过滤无效数据）
    valid_dataset = []
    for data in data_list:
        # 校验是否包含 translation 嵌套字段
        if "translation" not in data or not isinstance(data["translation"], dict):
            format_error_count += 1
            continue
        
        # 提取英德文本（去重、过滤空文本）
        en_text = data["translation"].get("en", "").strip()
        de_text = data["translation"].get("de", "").strip()
        
        # 过滤条件：文本非空 + 长度在合理范围（2~120字符）
        if not (en_text and de_text):
            format_error_count += 1
            continue
        if len(en_text) < 2 or len(en_text) > 120 or len(de_text) < 2 or len(de_text) > 120:
            format_error_count += 1
            continue
        
        # 保留原始格式，同时新增平级字段（兼容分词器和模型）
        valid_dataset.append({
            "translation": {"en": en_text, "de": de_text},
            "source": en_text if config.SRC_LANG == "en" else de_text,
            "target": de_text if config.TGT_LANG == "de" else en_text
        })

    # 截取样本数量（按配置的训练/验证集大小）
    max_sample_size = config.VAL_SIZE if is_val else config.TRAIN_SIZE
    if len(valid_dataset) > max_sample_size:
        valid_dataset = valid_dataset[:max_sample_size]
        print(f"   数据集过大，截取前 {max_sample_size} 条有效样本")

    # 输出加载统计
    print(f"✅ 本地数据集加载完成：")
    print(f"   - 原始数据：{total_lines} 条")
    print(f"   - 有效数据：{len(valid_dataset)} 条")
    print(f"   - 解析错误：{parse_error_count} 条（JSON格式错误）")
    print(f"   - 格式错误：{format_error_count} 条（缺少字段/空文本/长度异常）")

    return valid_dataset


def _load_hf_raw_dataset(config):
    """加载Hugging Face公开原始数据集（用于分词器训练）"""
    print(f"\n📥 加载公开数据集：{config.DATASET_NAME}-{config.DATASET_CONFIG}")
    try:
        # 加载训练集全量数据（保证分词器语料覆盖度）
        dataset = load_dataset(
            config.DATASET_NAME,
            config.DATASET_CONFIG,
            split="train",
            trust_remote_code=True,
            cache_dir=os.path.join("data", "cache")  # 缓存路径，避免重复下载
        )
    except Exception as e:
        raise RuntimeError(f"❌ 公开数据集加载失败：{str(e)}（请检查数据集名称/配置是否正确）") from e

    # 截取样本（加速分词器训练）
    max_sample_size = min(len(dataset), config.TRAIN_SIZE * 2)  # 取2倍训练集大小
    dataset = dataset.select(range(max_sample_size))

    print(f"✅ 公开数据集加载完成：{len(dataset)} 条样本")
    return dataset


class TranslationDataset(Dataset):
    """翻译数据集类（适配两种格式，用于模型训练/评估）"""
    def __init__(self, raw_data, tokenizer, config, is_train=True):
        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.max_length = config.MAX_LENGTH
        self.src_lang = config.SRC_LANG
        self.tgt_lang = config.TGT_LANG
        self.is_train = is_train

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        data = self.raw_data[idx]

        # 提取源文本和目标文本（适配两种格式）
        if "translation" in data:
            # 格式1：{"translation": {"en": "", "de": ""}}（本地数据集）
            src_text = data["translation"][self.src_lang].strip()
            tgt_text = data["translation"][self.tgt_lang].strip()
        elif "source" in data and "target" in data:
            # 格式2：{"source": "", "target": ""}（兼容公开数据集）
            src_text = data["source"].strip()
            tgt_text = data["target"].strip()
        else:
            src_text = ""
            tgt_text = ""

        # 编码文本（适配分词器的encode接口）
        src_ids = self.tokenizer.encode(
            src_text,
            max_length=self.max_length,
            padding=False,
            truncation=True
        )[0]  # encode返回列表，取单句结果
        tgt_ids = self.tokenizer.encode(
            tgt_text,
            max_length=self.max_length,
            padding=False,
            truncation=True
        )[0]

        # 返回模型所需的tensor格式
        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
            "src_text": src_text,
            "tgt_text": tgt_text
        }


def collate_fn(batch, pad_token_id):
    """批量处理函数（padding到批次内最大长度，适配模型输入）"""
    # 提取批次中的所有字段
    src_ids = [item["src_ids"] for item in batch]
    tgt_ids = [item["tgt_ids"] for item in batch]
    src_texts = [item["src_text"] for item in batch]
    tgt_texts = [item["tgt_text"] for item in batch]

    # 对source和target进行padding（用pad_token_id填充）
    src_ids_padded = torch.nn.utils.rnn.pad_sequence(
        src_ids, batch_first=True, padding_value=pad_token_id
    )
    tgt_ids_padded = torch.nn.utils.rnn.pad_sequence(
        tgt_ids, batch_first=True, padding_value=pad_token_id
    )

    return {
        "src_ids": src_ids_padded,
        "tgt_ids": tgt_ids_padded,
        "src_texts": src_texts,
        "tgt_texts": tgt_texts
    }


def get_data_loaders(config, tokenizer):
    """获取训练/验证数据加载器（用于模型训练）"""
    print("\n📥 构建数据加载器...")
    # 加载原始数据（根据配置选择本地/公开数据集）
    if config.USE_LOCAL_DATA:
        train_raw = _load_local_raw_dataset(config, is_val=False)
        val_raw = _load_local_raw_dataset(config, is_val=True)
    else:
        train_raw, val_raw = _load_hf_encoded_dataset(config)

    # 构建编码后的数据集（适配模型输入）
    train_dataset = TranslationDataset(train_raw, tokenizer, config, is_train=True)
    val_dataset = TranslationDataset(val_raw, tokenizer, config, is_train=False)

    # 构建DataLoader（批量加载+多线程预处理）
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,  # 训练集打乱
        num_workers=config.NUM_WORKERS,
        collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id),
        pin_memory=True,  # 加速GPU数据传输
        drop_last=True  # 丢弃最后一个不完整批次
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE * 2,  # 验证集批次加倍，加速评估
        shuffle=False,  # 验证集不打乱
        num_workers=config.NUM_WORKERS,
        collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id),
        pin_memory=True
    )

    # 输出数据加载器统计
    print(f"✅ 数据加载器构建完成：")
    print(f"   - 训练集：{len(train_dataset)} 条样本，{len(train_loader)} 个批次")
    print(f"   - 验证集：{len(val_dataset)} 条样本，{len(val_loader)} 个批次")
    print(f"   - 批次大小：训练集 {config.BATCH_SIZE}，验证集 {config.BATCH_SIZE * 2}")

    return train_loader, val_loader


def _load_hf_encoded_dataset(config):
    """加载Hugging Face公开数据集（用于模型训练/验证）"""
    print(f"\n📥 加载公开训练/验证数据集：{config.DATASET_NAME}-{config.DATASET_CONFIG}")
    try:
        dataset = load_dataset(
            config.DATASET_NAME,
            config.DATASET_CONFIG,
            splits=["train", "validation"],
            trust_remote_code=True,
            cache_dir=os.path.join("data", "cache")
        )
    except Exception as e:
        raise RuntimeError(f"❌ 公开数据集加载失败：{str(e)}") from e

    # 截取样本（控制数据规模，加速训练）
    train_raw = dataset["train"].select(range(min(len(dataset["train"]), config.TRAIN_SIZE)))
    val_raw = dataset["validation"].select(range(min(len(dataset["validation"]), config.VAL_SIZE)))

    return train_raw, val_raw