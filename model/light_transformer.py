import torch
import torch.nn as nn
from .positional_encoding import PositionalEncoding
from .attention import MultiHeadAttention, FlashAttentionWrapper


class TransformerEncoderLayerWithFlash(nn.Module):
    """支持Flash Attention的Encoder层"""

    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1,
                 activation='gelu', use_flash_attention=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout, use_flash_attention)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

    def forward(self, src, src_mask=None):
        # 自注意力
        src2 = self.self_attn(src, src, src, mask=src_mask)
        src = src + self.dropout(src2)
        src = self.norm1(src)

        # 前馈网络
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)

        return src


class LightTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 动态决定是否使用Flash Attention
        use_flash = FlashAttentionWrapper.should_use_flash_attention(
            config.MAX_LENGTH, threshold=128
        ) and config.USE_FLASH_ATTENTION

        print(f"使用Flash Attention: {use_flash}")

        # Encoder
        self.encoder_embedding = nn.Embedding(config.VOCAB_SIZE, config.D_MODEL)
        self.encoder_pos = PositionalEncoding(config.D_MODEL, config.MAX_LENGTH)

        # 使用自定义Encoder层支持Flash Attention
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayerWithFlash(
                d_model=config.D_MODEL,
                nhead=config.NHEAD,
                dim_feedforward=config.DIM_FEEDFORWARD,
                dropout=config.DROPOUT,
                use_flash_attention=use_flash
            ) for _ in range(config.NUM_ENCODER_LAYERS)
        ])

        # Decoder
        self.decoder_embedding = nn.Embedding(config.VOCAB_SIZE, config.D_MODEL)
        self.decoder_pos = PositionalEncoding(config.D_MODEL, config.MAX_LENGTH)

        # Decoder层（简化实现，使用标准TransformerDecoderLayer）
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.D_MODEL,
            nhead=config.NHEAD,
            dim_feedforward=config.DIM_FEEDFORWARD,
            dropout=config.DROPOUT,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, config.NUM_DECODER_LAYERS)

        # 输出层
        self.fc = nn.Linear(config.D_MODEL, config.VOCAB_SIZE)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encoder前向
        src_emb = self.encoder_embedding(src)
        src_emb = self.encoder_pos(src_emb)

        # 逐层处理Encoder
        enc_out = src_emb
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_mask=src_mask)

        # Decoder前向
        tgt_emb = self.decoder_embedding(tgt)
        tgt_emb = self.decoder_pos(tgt_emb)

        # 生成causal mask
        if tgt_mask is None:
            tgt_len = tgt.size(1)
            tgt_mask = self._generate_square_subsequent_mask(tgt_len).to(tgt.device)

        dec_out = self.decoder(tgt_emb, enc_out, tgt_mask=tgt_mask)

        # 输出预测
        logits = self.fc(dec_out)
        return logits

    def _generate_square_subsequent_mask(self, sz):
        """生成因果mask"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
