"""
PyTorch Dataset & DataLoader cho Text-to-Triples (hỗ trợ Curriculum Learning: 1 -> 2 -> >=3 triples).
Owner: Doãn Duy Lợi (24021549)
"""
from torch.utils.data import Dataset


class WebNLGDataset(Dataset):
    """Dataset wrapper cho WebNLG+ 2020 đã chuẩn hóa."""

    def __init__(self, data_path: str, tokenizer=None, max_length: int = 256):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]
