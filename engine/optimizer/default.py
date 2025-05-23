HYPER_DICT = {
    'partial': {
        'optim': "adamw",
        'lr': [0.00001, 0.000001, 0.0000001],
        'weight_decay': [0.0, 0.001, 0.00001],
        'lr_scheduler': "cosine",
        'batch_size': [8],
        'max_iter': [12800],
        'warmup_iter': 50,
        'warmup_type': "linear",
        'warmup_min_lr': 1e-5,
    },
    'adapter': {
        'optim': "adamw",
        'lr': [0.0001, 0.00001],
        'weight_decay': [0.0, 0.001, 0.00001],
        'lr_scheduler': "cosine",
        'batch_size': [8],
        'max_iter': [12800],
        'warmup_iter': 50,
        'warmup_type': "linear",
        'warmup_min_lr': 1e-5,
    }
}