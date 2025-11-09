import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from scripts.inference import TransformerInference


def evaluate_model():
    checkpoint_path = "outputs/checkpoints/latest_transformer.pth"
    inference = TransformerInference(checkpoint_path)

    # 评估集样例
    eval_sentences = [
        "Good morning!",
        "I love reading books.",
        "Can you help me?",
        "This is a great movie.",
        "How old are you?",
        "Where is the nearest restaurant?",
        "What time is it?",
        "Thank you very much.",
        "I don't understand.",
        "See you tomorrow."
    ]

    print("模型评估结果")
    print("=" * 50)

    for i, sentence in enumerate(eval_sentences, 1):
        print(f"样例 {i}:")
        print(f"输入: {sentence}")
        result = inference.beam_search_decode(sentence, beam_size=5)
        print(f"输出: {result}")
        print()


if __name__ == "__main__":
    evaluate_model()