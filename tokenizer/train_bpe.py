"""
Huấn luyện Custom BPE Tokenizer CHỈ trên tập train split của WebNLG+ 2020.
Owner: Doãn Duy Lợi (24021549)
"""


def train_bpe_tokenizer(
    train_texts: list[str],
    vocab_size: int = 16000,
    save_dir: str = "tokenizer/artifacts",
) -> None:
    raise NotImplementedError("Implement BPE tokenizer training using Hugging Face tokenizers.")


if __name__ == "__main__":
    pass
