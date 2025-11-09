import torch.optim as optim


def get_optimizer_and_scheduler(model, config):
    """获取优化器和学习率调度器"""
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.98),
        eps=1e-9
    )

    # 学习率调度器：线性热身然后线性衰减
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(
            (step + 1) / (config.NUM_WARMUP_STEPS + 1),
            (config.NUM_WARMUP_STEPS + 1) / (step + 1)
        )
    )

    return optimizer, scheduler
