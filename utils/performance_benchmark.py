import time
import torch
from model.attention import FlashAttentionWrapper


def benchmark_attention():
    """比较标准注意力和Flash Attention的性能"""
    print("注意力机制性能对比")
    print("=" * 50)

    seq_lengths = [64, 128, 256, 512]
    batch_size = 32
    d_model = 256
    nhead = 4

    for seq_len in seq_lengths:
        print(f"\n序列长度: {seq_len}")

        # 显存使用分析
        memory_info = FlashAttentionWrapper.memory_usage_comparison(
            seq_len, batch_size, nhead, d_model
        )
        print(f"标准注意力显存: {memory_info['standard_attention_MB']:.2f} MB")
        if FlashAttentionWrapper.is_available():
            print(f"Flash Attention显存: {memory_info['flash_attention_MB']:.2f} MB")
            print(f"显存减少: {memory_info['reduction_ratio'] * 100:.1f}%")
        else:
            print("Flash Attention: 不可用")

        # 性能建议
        recommended = FlashAttentionWrapper.should_use_flash_attention(seq_len)
        print(f"推荐使用Flash Attention: {recommended}")


if __name__ == "__main__":
    benchmark_attention()