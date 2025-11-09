from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.normalizers import NFKC
import os
import json
import warnings

def build_tokenizer(config, dataset=None):
    """构建基于BPE的分词器（优化版：强制用真实数据集训练，词汇表大小达标）"""
    # 1. 配置分词器保存路径（从config读取，确保训练/评估一致）
    tokenizer_path = getattr(config, "TOKENIZER_PATH", "outputs/tokenizer.json")
    os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)

    # 2. 优先加载已训练的分词器（若存在且词汇表大小达标）
    if os.path.exists(tokenizer_path):
        tokenizer = Tokenizer.from_file(tokenizer_path)
        vocab_size = tokenizer.get_vocab_size()
        # 检查词汇表大小是否合理（翻译任务至少8192）
        if vocab_size >= 8192:
            print(f"✅ 加载已有分词器：{tokenizer_path}（词汇表大小：{vocab_size}）")
            return tokenizer
        else:
            warnings.warn(f"⚠️  已存在的分词器词汇表过小（{vocab_size}），将重新训练")
            os.remove(tokenizer_path)  # 删除小词汇表分词器

    # 3. 训练新分词器：强制要求传入真实数据集（禁止用示例文本）
    print("📥 开始训练新分词器...")
    if dataset is None:
        raise ValueError("❌ 训练分词器必须传入真实数据集！请在train.py中传入数据集给build_tokenizer")

    # 4. 初始化分词器（新增归一化器，优化文本处理）
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = NFKC()  # 统一字符格式（如全角转半角、Unicode标准化）
    tokenizer.pre_tokenizer = Whitespace()  # 按空格分词（适合英/德等语言，中文需换用Char预分词）

    # 5. 配置BPE训练器（优化参数）
    trainer = BpeTrainer(
        vocab_size=config.VOCAB_SIZE,  # 从config读取（建议8192或16384）
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],  # 特殊token顺序固定
        min_frequency=2,  # 保留：过滤出现次数<2的稀有词
        show_progress=True,  # 显示训练进度
    )

    # 6. 准备训练语料（从平行语料中提取源语言和目标语言文本）
    def get_training_corpus():
        batch_size = 1000  # 批量生成，避免内存占用过大
        batch = []
        for example in dataset:
            # 适配常见平行语料格式（根据你的数据集调整字段名）
            if 'translation' in example:
                # 格式1：Hugging Face公开数据集（如iwslt2017-en-de）
                src_text = example['translation'].get(config.SRC_LANG, "")  # 源语言（如en）
                tgt_text = example['translation'].get(config.TGT_LANG, "")  # 目标语言（如de）
            elif config.USE_LOCAL_DATA and 'source' in example and 'target' in example:
                # 格式2：本地平行语料（source/target字段）
                src_text = example['source']
                tgt_text = example['target']
            else:
                # 格式3：自定义字段（如en/de）
                src_text = example.get(config.SRC_LANG, "")
                tgt_text = example.get(config.TGT_LANG, "")
            
            # 过滤空文本，添加到训练语料
            if src_text.strip():
                batch.append(src_text.strip())
            if tgt_text.strip():
                batch.append(tgt_text.strip())
            
            # 批量yield，优化内存
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    # 7. 训练分词器（用真实平行语料）
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
    final_vocab_size = tokenizer.get_vocab_size()
    if final_vocab_size < config.VOCAB_SIZE:
        warnings.warn(f"⚠️  训练语料词汇量不足，实际词汇表大小（{final_vocab_size}）< 配置值（{config.VOCAB_SIZE}）")

    # 8. 设置后处理模板（添加BOS/EOS，适配模型输入）
    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",  # 单句：BOS + 文本 + EOS
        pair="[BOS] $A [EOS] $B [EOS]",  # 成对文本（如翻译）：BOS + 源文本 + EOS + 目标文本 + EOS
        special_tokens=[
            ("[BOS]", tokenizer.token_to_id("[BOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ],
    )

    # 9. 保存分词器和词汇表信息
    tokenizer.save(tokenizer_path)
    vocab_info = {
        "vocab_size": final_vocab_size,
        "special_tokens": {
            "pad": tokenizer.token_to_id("[PAD]"),
            "unk": tokenizer.token_to_id("[UNK]"),
            "bos": tokenizer.token_to_id("[BOS]"),
            "eos": tokenizer.token_to_id("[EOS]")
        },
        "config_vocab_size": config.VOCAB_SIZE,
        "min_frequency": 2
    }
    with open(os.path.join(os.path.dirname(tokenizer_path), "tokenizer_info.json"), "w", encoding="utf-8") as f:
        json.dump(vocab_info, f, indent=2, ensure_ascii=False)

    print(f"🎉 分词器训练完成！保存路径：{tokenizer_path}（词汇表大小：{final_vocab_size}）")
    return tokenizer

class TokenizerWrapper:
    """分词器包装类，提供统一接口（适配训练/评估脚本）"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # 绑定特殊token ID（确保和模型一致）
        self.pad_token_id = tokenizer.token_to_id("[PAD]")
        self.bos_token_id = tokenizer.token_to_id("[BOS]")
        self.eos_token_id = tokenizer.token_to_id("[EOS]")
        self.unk_token_id = tokenizer.token_to_id("[UNK]")
        # 校验特殊token是否存在
        assert all(id is not None for id in [self.pad_token_id, self.bos_token_id, self.eos_token_id, self.unk_token_id]), \
            "❌ 分词器缺少必要的特殊token！"

    def encode(self, text, max_length=None, padding=True, truncation=True):
        """编码文本为token IDs（适配批量/单句输入）"""
        if isinstance(text, str):
            text = [text]  # 统一为列表格式
        
        # 根据tokenizers版本兼容性处理参数
        encode_kwargs = {}
        
        # 新版本tokenizers使用add_special_tokens而不是padding/truncation
        if hasattr(self.tokenizer, 'enable_padding'):
            if padding:
                self.tokenizer.enable_padding(
                    pad_id=self.pad_token_id, 
                    pad_token="[PAD]",
                    length=max_length
                )
            else:
                self.tokenizer.no_padding()
                
            if truncation and max_length:
                self.tokenizer.enable_truncation(max_length)
            else:
                self.tokenizer.no_truncation()
        else:
            # 旧版本兼容
            encode_kwargs = {
                'padding': padding,
                'truncation': truncation,
                'max_length': max_length
            }
        
        # 编码（支持批量处理）
        try:
            encodings = self.tokenizer.encode_batch(text, **encode_kwargs)
        except TypeError as e:
            # 如果仍然报错，回退到逐条编码
            print(f"⚠️  批量编码失败，使用逐条编码: {e}")
            encodings = []
            for t in text:
                try:
                    encoding = self.tokenizer.encode(t, **encode_kwargs)
                    encodings.append(encoding)
                except Exception as single_e:
                    print(f"❌ 单条编码失败: {single_e}")
                    # 返回空编码作为兜底
                    empty_encoding = type('EmptyEncoding', (), {'ids': [self.pad_token_id]})()
                    encodings.append(empty_encoding)
        
        # 转换为模型需要的格式（返回ids列表，适配Tensor转换）
        return [encoding.ids for encoding in encodings]

    def decode(self, token_ids):
        """解码token IDs为文本（自动过滤特殊token）"""
        # 处理批量输入
        if isinstance(token_ids[0], list):
            return [self._decode_single(ids) for ids in token_ids]
        return self._decode_single(token_ids)

    def _decode_single(self, token_ids):
        """解码单个句子的token IDs"""
        filtered_ids = [id for id in token_ids if id not in [self.pad_token_id, self.bos_token_id, self.eos_token_id]]
        return self.tokenizer.decode(filtered_ids, skip_special_tokens=False)

    def token_to_id(self, token):
        return self.tokenizer.token_to_id(token)

    def id_to_token(self, id):
        return self.tokenizer.id_to_token(id)

    def get_vocab_size(self):
        """获取词汇表大小（训练时更新config.VOCAB_SIZE）"""
        return self.tokenizer.get_vocab_size()

class SimpleEncoding:
    """简化的编码结果（兼容原有代码逻辑）"""
    def __init__(self, ids):
        self.ids = ids