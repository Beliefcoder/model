import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import time
import os


class TransformerTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = torch.device(config.DEVICE)

        self.scaler = GradScaler() if config.MIXED_PRECISION else None
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding
        self.best_val_loss = float('inf')

        # 创建输出目录
        os.makedirs(os.path.dirname(config.CHECKPOINT_PATH), exist_ok=True)
        os.makedirs(config.LOG_DIR, exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_steps = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            start_time = time.time()

            src = batch['src_ids'].to(self.device, non_blocking=True)
            tgt = batch['tgt_ids'].to(self.device, non_blocking=True)

            # 准备decoder输入和标签 (shift-right)
            tgt_input = tgt[:, :-1]
            tgt_labels = tgt[:, 1:]

            self.optimizer.zero_grad()

            # 混合精度训练
            if self.config.MIXED_PRECISION:
                with autocast():
                    logits = self.model(src, tgt_input)
                    loss = self.criterion(
                        logits.reshape(-1, logits.size(-1)),
                        tgt_labels.reshape(-1)
                    )

                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(src, tgt_input)
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_labels.reshape(-1)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP)
                self.optimizer.step()

            self.scheduler.step()

            total_loss += loss.item()
            batch_time = time.time() - start_time

            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f'Epoch [{epoch + 1}/{self.config.NUM_EPOCHS}], '
                      f'Step [{batch_idx + 1}/{total_steps}], '
                      f'Loss: {avg_loss:.4f}, '
                      f'LR: {self.scheduler.get_last_lr()[0]:.2e}, '
                      f'Time: {batch_time:.3f}s')

        return total_loss / total_steps

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in self.val_loader:
                src = batch['src_ids'].to(self.device, non_blocking=True)
                tgt = batch['tgt_ids'].to(self.device, non_blocking=True)

                tgt_input = tgt[:, :-1]
                tgt_labels = tgt[:, 1:]

                if self.config.MIXED_PRECISION:
                    with autocast():
                        logits = self.model(src, tgt_input)
                        loss = self.criterion(
                            logits.reshape(-1, logits.size(-1)),
                            tgt_labels.reshape(-1)
                        )
                else:
                    logits = self.model(src, tgt_input)
                    loss = self.criterion(
                        logits.reshape(-1, logits.size(-1)),
                        tgt_labels.reshape(-1)
                    )

                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        print(f'Validation - Epoch [{epoch + 1}], '
              f'Loss: {avg_loss:.4f}, '
              f'Perplexity: {perplexity:.2f}')

        return avg_loss, perplexity

    def train(self):
        print("开始训练...")
        print(f"设备: {self.device}")
        print(f"训练样本: {len(self.train_loader.dataset)}")
        print(f"验证样本: {len(self.val_loader.dataset)}")
        print(f"参数数量: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(self.config.NUM_EPOCHS):
            epoch_start = time.time()

            # 训练一个epoch
            train_loss = self.train_epoch(epoch)

            # 验证
            val_loss, val_ppl = self.validate(epoch)

            epoch_time = time.time() - epoch_start

            print(f'Epoch [{epoch + 1}] 完成, '
                  f'时间: {epoch_time:.2f}s, '
                  f'训练损失: {train_loss:.4f}, '
                  f'验证损失: {val_loss:.4f}, '
                  f'验证困惑度: {val_ppl:.2f}')

            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.config.CHECKPOINT_PATH)
                print(f'保存最佳模型，验证损失: {val_loss:.4f}')

            print('-' * 60)

        print("训练完成!")