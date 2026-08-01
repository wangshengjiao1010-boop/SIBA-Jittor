class args:
    # trainset path
    ir_path = "/root/autodl-tmp/datasets/SIBA/train/ir"
    vi_path = "/root/autodl-tmp/datasets/SIBA/train/vi"
    patch_size = 128

    model_save_path = "./checkpoint/test"
    use_gpu_number = "0"
    middle_channel = 48

    epochs = 60
    optim_step = 25

    batch_size = 4
    init_lr = 1e-4
    use_gpu = True
    optim_gamma = 0.5
    weight_decay = 0
