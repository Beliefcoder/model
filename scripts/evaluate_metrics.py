#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.meteor_score import meteor_score
import nltk
import json
import os

class TranslationEvaluator:
    """翻译质量评估器"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # 确保nltk数据已下载
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def calculate_bleu(self, references, hypotheses):
        """计算BLEU分数"""
        # 单个句子BLEU
        sentence_bleu_scores = []
        for ref, hyp in zip(references, hypotheses):
            ref_tokens = [ref.split()]
            hyp_tokens = hyp.split()
            score = sentence_bleu(ref_tokens, hyp_tokens)
            sentence_bleu_scores.append(score)
        
        # 语料库BLEU
        ref_corpus = [[ref.split()] for ref in references]
        hyp_corpus = [hyp.split() for hyp in hypotheses]
        corpus_bleu_score = corpus_bleu(ref_corpus, hyp_corpus)
        
        return {
            "sentence_bleu_mean": np.mean(sentence_bleu_scores),
            "sentence_bleu_std": np.std(sentence_bleu_scores),
            "corpus_bleu": corpus_bleu_score,
            "bleu_1": corpus_bleu(ref_corpus, hyp_corpus, weights=(1, 0, 0, 0)),
            "bleu_2": corpus_bleu(ref_corpus, hyp_corpus, weights=(0.5, 0.5, 0, 0)),
            "bleu_3": corpus_bleu(ref_corpus, hyp_corpus, weights=(0.33, 0.33, 0.33, 0)),
            "bleu_4": corpus_bleu(ref_corpus, hyp_corpus, weights=(0.25, 0.25, 0.25, 0.25))
        }
    
    def calculate_meteor(self, references, hypotheses):
        """计算METEOR分数"""
        meteor_scores = []
        for ref, hyp in zip(references, hypotheses):
            score = meteor_score([ref], hyp)
            meteor_scores.append(score)
        
        return {
            "meteor_mean": np.mean(meteor_scores),
            "meteor_std": np.std(meteor_scores)
        }
    
    def calculate_rouge(self, references, hypotheses):
        """计算ROUGE分数（简化版）"""
        # 这里实现ROUGE-L的简化版本
        rouge_l_scores = []
        
        for ref, hyp in zip(references, hypotheses):
            ref_words = ref.split()
            hyp_words = hyp.split()
            
            # 计算最长公共子序列
            lcs_length = self.longest_common_subsequence(ref_words, hyp_words)
            
            if len(ref_words) == 0 or len(hyp_words) == 0:
                rouge_l_scores.append(0.0)
                continue
            
            precision = lcs_length / len(hyp_words)
            recall = lcs_length / len(ref_words)
            
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            
            rouge_l_scores.append(f1)
        
        return {
            "rouge_l_mean": np.mean(rouge_l_scores),
            "rouge_l_std": np.std(rouge_l_scores)
        }
    
    def longest_common_subsequence(self, seq1, seq2):
        """计算最长公共子序列长度"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def calculate_perplexity(self, model, data_loader, device):
        """计算困惑度"""
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in data_loader:
                src = batch['src_ids'].to(device)
                tgt = batch['tgt_ids'].to(device)
                
                tgt_input = tgt[:, :-1]
                tgt_labels = tgt[:, 1:]
                
                logits = model(src, tgt_input)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), 
                    tgt_labels.reshape(-1),
                    reduction='sum'
                )
                
                total_loss += loss.item()
                total_tokens += (tgt_labels != 0).sum().item()  # 忽略padding
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity
    
    def calculate_accuracy(self, model, data_loader, device):
        """计算准确率"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in data_loader:
                src = batch['src_ids'].to(device)
                tgt = batch['tgt_ids'].to(device)
                
                tgt_input = tgt[:, :-1]
                tgt_labels = tgt[:, 1:]
                
                logits = model(src, tgt_input)
                predictions = logits.argmax(dim=-1)
                
                # 只计算非padding位置的准确率
                mask = (tgt_labels != 0)
                correct += ((predictions == tgt_labels) & mask).sum().item()
                total += mask.sum().item()
        
        accuracy = correct / total if total > 0 else 0
        return accuracy

def comprehensive_evaluation(model, data_loader, tokenizer, device, num_samples=100):
    """综合评估模型性能"""
    evaluator = TranslationEvaluator(tokenizer)
    
    print("开始综合评估...")
    
    # 1. 计算困惑度
    perplexity = evaluator.calculate_perplexity(model, data_loader, device)
    print(f"困惑度: {perplexity:.2f}")
    
    # 2. 计算准确率
    accuracy = evaluator.calculate_accuracy(model, data_loader, device)
    print(f"准确率: {accuracy:.3f}")
    
    # 3. 生成翻译并计算其他指标
    references = []
    hypotheses = []
    
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= num_samples:  # 限制样本数量
                break
                
            src = batch['src_ids'].to(device)
            src_texts = batch['src_texts']
            tgt_texts = batch['tgt_texts']
            
            # 使用贪心搜索生成翻译
            generated_translations = greedy_decode_batch(model, src, tokenizer, device)
            
            references.extend(tgt_texts)
            hypotheses.extend(generated_translations)
    
    # 4. 计算BLEU分数
    bleu_metrics = evaluator.calculate_bleu(references, hypotheses)
    print(f"语料库BLEU: {bleu_metrics['corpus_bleu']:.3f}")
    print(f"BLEU-1: {bleu_metrics['bleu_1']:.3f}, BLEU-2: {bleu_metrics['bleu_2']:.3f}")
    print(f"BLEU-3: {bleu_metrics['bleu_3']:.3f}, BLEU-4: {bleu_metrics['bleu_4']:.3f}")
    
    # 5. 计算METEOR分数
    meteor_metrics = evaluator.calculate_meteor(references, hypotheses)
    print(f"METEOR: {meteor_metrics['meteor_mean']:.3f}")
    
    # 6. 计算ROUGE分数
    rouge_metrics = evaluator.calculate_rouge(references, hypotheses)
    print(f"ROUGE-L: {rouge_metrics['rouge_l_mean']:.3f}")
    
    # 保存评估结果
    results = {
        "perplexity": perplexity,
        "accuracy": accuracy,
        "bleu_metrics": bleu_metrics,
        "meteor_metrics": meteor_metrics,
        "rouge_metrics": rouge_metrics,
        "sample_translations": [
            {
                "source": src,
                "reference": ref,
                "hypothesis": hyp
            }
            for src, ref, hyp in zip(src_texts[:5], references[:5], hypotheses[:5])
        ]
    }
    
    # 保存结果
    os.makedirs("results", exist_ok=True)
    with open("results/comprehensive_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results

def greedy_decode_batch(model, src, tokenizer, device, max_length=64):
    """批量贪心解码"""
    batch_size = src.size(0)
    translations = []
    
    for i in range(batch_size):
        src_single = src[i:i+1]
        translation = greedy_decode_single(model, src_single, tokenizer, device, max_length)
        translations.append(translation)
    
    return translations

def greedy_decode_single(model, src, tokenizer, device, max_length=64):
    """单个句子贪心解码"""
    model.eval()
    
    # 编码器前向传播
    with torch.no_grad():
        src_emb = model.encoder_embedding(src)
        src_emb = model.encoder_pos(src_emb)
        enc_output = model.encoder(src_emb)
    
    # 初始化解码器输入
    tgt = torch.tensor([[tokenizer.bos_token_id]], device=device)
    
    for i in range(max_length):
        with torch.no_grad():
            tgt_emb = model.decoder_embedding(tgt)
            tgt_emb = model.decoder_pos(tgt_emb)
            dec_output = model.decoder(tgt_emb, enc_output)
            logits = model.fc(dec_output[:, -1, :])
            next_token = logits.argmax(dim=-1, keepdim=True)
        
        tgt = torch.cat([tgt, next_token], dim=1)
        
        # 如果生成了EOS token，停止生成
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    # 解码为文本
    translation = tokenizer.decode(tgt.squeeze().tolist())
    return translation

#def main():
#    """运行综合评估"""
#    import sys
#    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
#    
#    from config.training_config import TrainingConfig
#    from data.tokenizer import build_tokenizer
#    from data.dataset_loader import get_data_loaders
#    from model.light_transformer import LightTransformer
#    
#    # 加载配置和模型
#    config = TrainingConfig()
#    tokenizer = build_tokenizer(config)
#    
#    # 加载数据
#    _, val_loader = get_data_loaders(config, tokenizer)
#    
#    # 加载模型
#    device = torch.device(config.DEVICE)
#    model = LightTransformer(config).to(device)
#    
#    # 加载训练好的权重
#    checkpoint_path = "outputs/checkpoints/latest_transformer.pth"
#    if os.path.exists(checkpoint_path):
#        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
#        print(f"加载模型权重: {checkpoint_path}")
#    else:
#        print("警告: 未找到训练好的模型权重，使用随机初始化的模型")
#    
#    # 运行综合评估
#    results = comprehensive_evaluation(model, val_loader, tokenizer, device)
#    
#    print("\n综合评估完成!")
#    print(f"主要指标:")
#    print(f"  困惑度: {results['perplexity']:.2f}")
#    print(f"  准确率: {results['accuracy']:.3f}")
#    print(f"  BLEU-4: {results['bleu_metrics']['bleu_4']:.3f}")
#    print(f"  METEOR: {results['meteor_metrics']['meteor_mean']:.3f}")
#    print(f"  ROUGE-L: {results['rouge_metrics']['rouge_l_mean']:.3f}")
def main():
    """运行综合评估（强制匹配训练时的配置）"""
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
    # -------------------------- 关键1：匹配训练时的 Config 配置 --------------------------
    from config.training_config import TrainingConfig  # 若训练时用了其他Config，替换成对应的
    config = TrainingConfig()
    
    # 强制设置：和训练时的权重形状匹配（VOCAB_SIZE=36）
    config.VOCAB_SIZE = 36  # 关键！手动改成训练时的词汇表大小（36）
    config.USE_LOCAL_DATA = False  # 强制禁用本地数据，稳定加载公开数据集
    config.MAX_LENGTH = 64  # 保持和训练时一致（若训练时改了，这里同步改）
    
    # -------------------------- 关键2：加载依赖组件 --------------------------
    from data.tokenizer import build_tokenizer
    from data.dataset_loader import get_data_loaders
    from model.light_transformer import LightTransformer
    
    # 加载分词器（此时分词器的词汇表大小会和 config.VOCAB_SIZE 一致）
    tokenizer = build_tokenizer(config)
    
    # 稳定加载公开数据集（禁用本地数据，避免找 data/custom_dataset）
    _, val_loader = get_data_loaders(config, tokenizer)
    print(f"✅ 数据集加载成功：验证集大小 {len(val_loader.dataset)}")
    
    # -------------------------- 关键3：加载模型（修复权重匹配+安全警告） --------------------------
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    model = LightTransformer(config).to(device)  # 此时模型参数形状是 [36, 256]，和权重匹配
    
    checkpoint_path = "outputs/checkpoints/latest_transformer.pth"
    if os.path.exists(checkpoint_path):
        # 修复安全警告：添加 weights_only=True；权重形状已匹配，能正常加载
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        print(f"✅ 成功加载模型权重（词汇表大小 36，和训练时一致）")
    else:
        print("警告: 未找到训练好的模型权重，使用随机初始化的模型")
    
    # -------------------------- 运行评估 --------------------------
    results = comprehensive_evaluation(model, val_loader, tokenizer, device)
    
    print("\n🎉 综合评估完成!")
    print(f"主要指标:")
    print(f"  困惑度: {results['perplexity']:.2f}")
    print(f"  准确率: {results['accuracy']:.3f}")
    print(f"  BLEU-4: {results['bleu_metrics']['bleu_4']:.3f}")
    print(f"  METEOR: {results['meteor_metrics']['meteor_mean']:.3f}")
    print(f"  ROUGE-L: {results['rouge_metrics']['rouge_l_mean']:.3f}")
    print(f"评估结果已保存到: results/comprehensive_evaluation.json")

if __name__ == "__main__":
    main()