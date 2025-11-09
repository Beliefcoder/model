import torch
import torch.nn as nn
import math

try:
    from flash_attn import flash_attn_func

    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    print("Flash Attention not available, using standard attention")


class MultiHeadAttention(nn.Module):
    """支持Flash Attention的多头注意力模块"""

    def __init__(self, d_model, nhead, dropout=0.1, use_flash_attention=False):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.use_flash_attention = use_flash_attention and FLASH_ATTENTION_AVAILABLE

        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, query, key, value, mask=None, is_causal=False):
        batch_size, seq_len, _ = query.size()

        # 线性变换
        Q = self.w_q(query)  # [batch_size, seq_len, d_model]
        K = self.w_k(key)  # [batch_size, seq_len, d_model]
        V = self.w_v(value)  # [batch_size, seq_len, d_model]

        # 重形状为多头 [batch_size, seq_len, nhead, head_dim]
        Q = Q.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        if self.use_flash_attention:
            # 使用Flash Attention
            attn_output = flash_attn_func(
                Q, K, V,
                dropout_p=self.dropout.p if self.training else 0.0,
                softmax_scale=self.scale,
                causal=is_causal
            )
        else:
            # 标准Scaled Dot-Product Attention
            attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

            # 应用mask
            if mask is not None:
                attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
            if is_causal:
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=Q.device),
                    diagonal=1
                ).bool()
                attn_weights = attn_weights.masked_fill(
                    causal_mask.unsqueeze(0).unsqueeze(0), -1e9
                )

            attn_weights = torch.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)

            attn_output = torch.matmul(attn_weights, V)

        # 合并多头 [batch_size, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        # 输出投影
        output = self.w_o(attn_output)
        return output


class FlashAttentionWrapper:
    """Flash Attention配置包装器"""

    @staticmethod
    def is_available():
        return FLASH_ATTENTION_AVAILABLE

    @staticmethod
    def should_use_flash_attention(seq_len, threshold=128):
        """
        判断是否应该使用Flash Attention
        """
        return FLASH_ATTENTION_AVAILABLE and seq_len >= threshold

    @staticmethod
    def memory_usage_comparison(seq_len, batch_size=32, nhead=8, d_model=512):
        """比较标准注意力和Flash Attention的显存使用"""
        # 标准注意力中间矩阵大小
        standard_memory = batch_size * nhead * seq_len * seq_len * 4  # 4 bytes per float

        # Flash Attention (分块计算，显著减少)
        block_size = min(64, seq_len)  # 典型分块大小
        flash_memory = batch_size * nhead * seq_len * block_size * 4 * 2  # 估算值

        return {
            'standard_attention_MB': standard_memory / (1024 * 1024),
            'flash_attention_MB': flash_memory / (1024 * 1024),
            'reduction_ratio': (standard_memory - flash_memory) / standard_memory
        }
