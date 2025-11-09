import torch
import torch.nn as nn
from typing import List, Tuple


class BeamSearchDecoder:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def beam_search(
            self,
            src_input: torch.Tensor,
            beam_size: int = 5,
            max_length: int = 64,
            length_penalty: float = 0.6,
            early_stopping: bool = True
    ) -> List[Tuple[torch.Tensor, float]]:
        """
        Beam Search解码实现
        """
        self.model.eval()

        with torch.no_grad():
            # 编码器前向传播
            src_emb = self.model.encoder_embedding(src_input)
            src_emb = self.model.encoder_pos(src_emb)

            enc_out = src_emb
            for layer in self.model.encoder_layers:
                enc_out = layer(enc_out)

            # 初始化束搜索
            start_token = torch.tensor([[self.tokenizer.token_to_id("<s>")]], device=self.device)
            eos_token = self.tokenizer.token_to_id("</s>")

            # 初始化候选序列: (序列tensor, 累积对数概率, 是否完成)
            sequences = [
                (start_token, 0.0, False)
            ]

            completed_sequences = []

            for step in range(max_length):
                all_candidates = []

                # 对每个候选序列扩展
                for seq, score, finished in sequences:
                    if finished:
                        all_candidates.append((seq, score, True))
                        continue

                    # 解码器前向传播
                    tgt_emb = self.model.decoder_embedding(seq)
                    tgt_emb = self.model.decoder_pos(tgt_emb)

                    # 生成causal mask
                    tgt_len = seq.size(1)
                    tgt_mask = self.model._generate_square_subsequent_mask(tgt_len).to(self.device)

                    decoder_output = self.model.decoder(tgt_emb, enc_out, tgt_mask=tgt_mask)
                    logits = self.model.fc(decoder_output[:, -1, :])  # 最后一个token的logits
                    log_probs = torch.log_softmax(logits, dim=-1)
                    topk_probs, topk_tokens = torch.topk(log_probs, beam_size, dim=-1)

                    # 生成新的候选序列
                    for i in range(beam_size):
                        next_token = topk_tokens[0, i].unsqueeze(0).unsqueeze(0)
                        next_seq = torch.cat([seq, next_token], dim=1)
                        next_score = score + topk_probs[0, i].item()

                        # 检查是否结束
                        is_finished = (next_token.item() == eos_token) or (step == max_length - 1)

                        all_candidates.append((next_seq, next_score, is_finished))

                # 选择top-k候选
                ordered = sorted(
                    all_candidates,
                    key=lambda x: x[1] / ((x[0].size(1) + 1) ** length_penalty),
                    reverse=True
                )
                sequences = ordered[:beam_size]

                # 收集完成的序列
                new_completed = [seq for seq in sequences if seq[2]]
                completed_sequences.extend(new_completed)
                sequences = [seq for seq in sequences if not seq[2]]

                # 提前停止条件
                if early_stopping and len(completed_sequences) >= beam_size:
                    break

                if len(sequences) == 0:
                    break

            # 如果没有完成的序列，使用当前最佳序列
            if not completed_sequences:
                completed_sequences = sequences

            # 应用长度惩罚并排序
            scored_sequences = []
            for seq, score, _ in completed_sequences:
                # 长度归一化得分
                length_penalized_score = score / (seq.size(1) ** length_penalty)
                scored_sequences.append((seq, length_penalized_score))

            # 按得分排序返回
            return sorted(scored_sequences, key=lambda x: x[1], reverse=True)

    def decode_sequence(self, sequence: torch.Tensor) -> str:
        """将token序列解码为文本"""
        tokens = sequence.squeeze().tolist()
        # 移除开始和结束token
        if tokens[0] == self.tokenizer.token_to_id("<s>"):
            tokens = tokens[1:]
        if tokens and tokens[-1] == self.tokenizer.token_to_id("</s>"):
            tokens = tokens[:-1]
        return self.tokenizer.decode(tokens)
