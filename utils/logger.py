import os
import datetime


class TrainingLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"training_{timestamp}.log")

    def log_step(self, epoch, step, loss):
        log_msg = f"Epoch [{epoch}], Step [{step}], Loss: {loss:.4f}"
        print(log_msg)
        with open(self.log_file, "a") as f:
            f.write(log_msg + "\n")

    def log_validation(self, epoch, loss, perplexity):
        log_msg = f"Epoch [{epoch}], Validation Loss: {loss:.4f}, Perplexity: {perplexity:.2f}"
        print(log_msg)
        with open(self.log_file, "a") as f:
            f.write(log_msg + "\n")

    def log_epoch(self, epoch, train_loss, val_loss, val_ppl):
        log_msg = f"Epoch [{epoch}] Summary: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}"
        print(log_msg)
        with open(self.log_file, "a") as f:
            f.write(log_msg + "\n")
