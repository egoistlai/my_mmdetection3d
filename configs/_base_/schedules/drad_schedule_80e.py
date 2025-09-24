# 文件: configs/_base_/schedules/drad_schedule_80e.py

# 训练总轮数
max_epochs = 80
# 训练样本总数 (根据您之前的日志)
total_samples = 4135
# 批次大小 (根据您的drad_dataset.py)
batch_size = 4
# 总迭代次数 (用于 OneCycleLR)
total_steps = max_epochs * (total_samples // batch_size + 1)

# 优化器封装 (AdamW with custom betas)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.001,  # OneCycleLR 的最大学习率
        betas=(0.95, 0.99), # 使用您示例中提供的 betas，通常更稳定
        weight_decay=0.01
    ),
    clip_grad=dict(max_norm=35, norm_type=2)
)

# OneCycle 学习率调度器 (完全兼容您框架的版本)
param_scheduler = [
    dict(
        type='OneCycleLR',
        by_epoch=False, # 强烈建议按迭代次数(iter)进行调度，更为精确
        total_steps=total_steps,
        eta_max=0.001,
        pct_start=0.4,
        div_factor=10,
        final_div_factor=100,
    )
]

# 运行配置
train_cfg = dict(
    by_epoch=True, # 训练流程按 epoch 组织
    max_epochs=max_epochs,
    val_interval=5
)
val_cfg = dict()
test_cfg = dict()