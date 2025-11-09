import torch
class MemoryMonitor:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def log_memory_usage(self):
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device) / 1024 ** 3
            reserved = torch.cuda.memory_reserved(self.device) / 1024 ** 3
            max_allocated = torch.cuda.max_memory_allocated(self.device) / 1024 ** 3
            print(f"GPU内存: {allocated:.2f}GB (当前) / {max_allocated:.2f}GB (峰值) / {reserved:.2f}GB (保留)")
