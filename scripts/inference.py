import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from config.training_config import TrainingConfig
from data.tokenizer import build_tokenizer
from model.light_transformer import LightTransformer
from model.beam_search import BeamSearchDecoder


class TransformerInference:
    def __init__(self, config_path=None):
        self.config = TrainingConfig()
        self.tokenizer = build_tokenizer(self.config)
        self.device = torch.device(self.config.DEVICE)

        # 加载模型
        self.model = LightTransformer(self.config).to(self.device)
        if config_path and os.path.exists(config_path):
            state_dict = torch.load(config_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"加载模型权重: {config_path}")
        else:
            print("使用随机初始化的模型")

        self.model.eval()
        self.beam_decoder = BeamSearchDecoder(self.model, self.tokenizer, self.device)

    def greedy_decode(self, text: str, max_length: int = 64) -> str:
        """贪心搜索解码"""
        # 编码输入
        encoding = self.tokenizer.encode(text)
        src_input = torch.tensor([encoding.ids], device=self.device)

        # 初始化输出序列
        output = torch.tensor([[self.tokenizer.token_to_id("<s>")]], device=self.device)

        with torch.no_grad():
            for i in range(max_length):
                logits = self.model(src_input, output)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                output = torch.cat([output, next_token], dim=1)

                # 检查是否结束
                if next_token.item() == self.tokenizer.token_to_id("</s>"):
                    break

        return self.beam_decoder.decode_sequence(output)

    def beam_search_decode(self, text: str, beam_size: int = 5, max_length: int = 64) -> str:
        """束搜索解码"""
        encoding = self.tokenizer.encode(text)
        src_input = torch.tensor([encoding.ids], device=self.device)

        # 执行束搜索
        results = self.beam_decoder.beam_search(
            src_input,
            beam_size=beam_size,
            max_length=max_length
        )

        # 返回最佳结果
        if results:
            best_sequence, best_score = results[0]
            return self.beam_decoder.decode_sequence(best_sequence)
        return ""

    def compare_decoding_methods(self, text: str):
        """比较不同解码方法的结果"""
        print(f"输入: {text}")

        # 贪心搜索
        greedy_result = self.greedy_decode(text)
        print(f"贪心搜索: {greedy_result}")

        # 束搜索
        beam_result = self.beam_search_decode(text, beam_size=5)
        print(f"束搜索 (k=5): {beam_result}")

        # 更大束宽
        beam_result_large = self.beam_search_decode(text, beam_size=10)
        print(f"束搜索 (k=10): {beam_result_large}")


def main():
    checkpoint_path = "outputs/checkpoints/latest_transformer.pth"
    inference = TransformerInference(checkpoint_path)

    # 测试样例
    test_sentences = [
        "Hello, how are you?",
        "What is your name?",
        "The weather is nice today.",
        "I love machine learning."
    ]

    print("推理测试")
    print("=" * 50)

    for sentence in test_sentences:
        inference.compare_decoding_methods(sentence)
        print("-" * 50)


if __name__ == "__main__":
    main()