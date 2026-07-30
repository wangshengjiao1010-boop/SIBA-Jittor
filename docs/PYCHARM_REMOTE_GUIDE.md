# PyCharm 远程开发与现场演示

本项目以 PyCharm 远程 SSH 为主要开发和演示方式。Jupyter Notebook 仅用于逐模块测试展示，不负责长期训练。

## 1. 两套 Conda 环境

新建环境：

```bash
cd /root/autodl-tmp/SIBA-Jittor
bash scripts/setup_envs.sh
bash scripts/complete_env_setup.sh
```

环境位于数据盘：

```text
/root/autodl-tmp/envs/PytorchDome
/root/autodl-tmp/envs/JittorDome
```

`/root/autodl-tmp/envs/JittorDome/bin/python` 不是额外编造的 Python，它就是 `JittorDome` Conda 环境中的解释器。PyCharm 远程解释器必须指向这个可执行文件。

激活方式：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate PytorchDome
python -c "import torch; print(torch.__version__)"

conda deactivate
conda activate JittorDome
python -c "import jittor as jt; print(jt.__version__)"
```

已经存在并完成实验的旧环境为：

```text
/root/autodl-tmp/envs/siba_torch
/root/autodl-tmp/envs/siba_jittor
```

若不想重新安装依赖，可在无卡模式运行：

```bash
bash scripts/clone_tested_envs_to_named.sh
```

该脚本从已验证环境克隆出 `PytorchDome` 和 `JittorDome`。所有正式脚本优先使用新名称；如果新名称尚不存在，会自动兼容旧环境。

## 2. PyCharm 连接 AutoDL

1. 在 AutoDL 控制台以无卡模式开机，复制当前 SSH 主机、端口和密码。
2. PyCharm 打开本项目，进入 `Settings | Project | Python Interpreter`。
3. 选择 `Add Interpreter | On SSH`，填写 AutoDL 主机、端口和 `root` 用户。
4. PyTorch 解释器选择 `/root/autodl-tmp/envs/PytorchDome/bin/python`。
5. 再添加一个 SSH 解释器，Jittor 选择 `/root/autodl-tmp/envs/JittorDome/bin/python`。
6. 远程项目目录固定为 `/root/autodl-tmp/SIBA-Jittor`。
7. 本地项目目录映射到该远程目录，不要映射到 `/root` 或 `/root/autodl-tmp/envs`。

PyCharm 每次只能给当前 Run Configuration 选择一个解释器。运行 PyTorch 时选择 `PytorchDome`，运行 Jittor 时选择 `JittorDome`，右下角或解释器设置中的显示名称才会随之变化。

## 3. 为什么不直接修改 `args_SIBA.py`

官方 `args/args_SIBA.py` 保留了占位路径。为了保持冻结的官方源码和 Jittor 镜像可逐文件审计，正式实验不直接改这两个文件，而由 `tools/run_training.py` 在运行时注入数据和输出路径。

因此在 PyCharm 中可以运行 `tools/run_training.py`，但不应为了方便而把个人绝对路径写进算法源码。

## 4. PyCharm 短任务配置

### PyTorch 训练入口检查

- Script path: `/root/autodl-tmp/SIBA-Jittor/tools/run_training.py`
- Python interpreter: `PytorchDome`
- Working directory: `/root/autodl-tmp/SIBA-Jittor`
- Parameters:

```text
--framework pytorch --ir-path /root/autodl-tmp/datasets/SIBA/train/ir --vi-path /root/autodl-tmp/datasets/SIBA/train/vi --output /root/autodl-tmp/SIBA-Jittor/checkpoints/pycharm_pytorch_check --epochs 60 --gpu-number 0 --seed 2025
```

### Jittor 训练入口检查

- Script path: `/root/autodl-tmp/SIBA-Jittor/tools/run_training.py`
- Python interpreter: `JittorDome`
- Working directory: `/root/autodl-tmp/SIBA-Jittor`
- Parameters:

```text
--framework jittor --ir-path /root/autodl-tmp/datasets/SIBA/train/ir --vi-path /root/autodl-tmp/datasets/SIBA/train/vi --output /root/autodl-tmp/SIBA-Jittor/checkpoints/pycharm_jittor_check --epochs 60 --gpu-number 0 --seed 2025
```

以上参数是正式 60 轮配置，不是缩水配置。现场演示不要重新启动完整训练；打开已经完成的日志和曲线即可。

## 5. 长训练必须使用 `screen -S kk`

PyCharm 的 SSH 或 Run 窗口关闭后，前台进程可能中断。完整训练使用 PyCharm 下方的远程 Terminal 执行：

```bash
cd /root/autodl-tmp/SIBA-Jittor
RUN_TAG=$(date +%Y%m%d_%H%M%S) bash scripts/train_full_sequence_screen.sh
screen -r kk
```

脱离但不停止训练：按 `Ctrl+A`，再按 `D`。

重新查看：

```bash
screen -ls
screen -r kk
```

查看正式训练日志：

```bash
tail -f logs/jittor_msrs_roadscene_60e_<RUN_TAG>/train.log
tail -f logs/pytorch_msrs_roadscene_60e_<RUN_TAG>/train.log
```

完整训练已完成，本项目不需要为了演示再次消耗 60 轮 GPU。

## 6. 推理演示

在 PyCharm 中选择 `JittorDome`，运行：

```bash
cd /root/autodl-tmp/SIBA-Jittor
python tools/run_inference.py \
  --framework jittor \
  --checkpoint checkpoints/jittor_msrs_roadscene_60e_20260727_siba_official_protocol/07-27-04-52/SIBA_epoch60.pkl \
  --data-dir /root/autodl-tmp/datasets/SIBA/test/TNO \
  --output results/pycharm_demo_jittor_tno \
  --use-cuda --warmup-runs 3 --timing-mode synchronized
```

运行后展示：

- `results/pycharm_demo_jittor_tno/summary.json`
- `results/pycharm_demo_jittor_tno/timing.csv`
- 新生成的 45 张 TNO 融合图

## 7. GitHub 同步

PyCharm、Jupyter 和 GitHub 没有绑定关系。训练结果只有经过 `git add`、`git commit`、`git push` 才会同步到 GitHub。

推荐先在 GitHub 创建空的公开仓库 `SIBA-Jittor`，不要初始化 README，然后在 PyCharm Terminal 执行：

```bash
cd /root/autodl-tmp/SIBA-Jittor
git remote add origin https://github.com/wangshengjiao1010-boop/SIBA-Jittor.git
git branch -M main
git push -u origin main
```

GitHub 已不支持账户密码执行 Git 推送。使用 Personal Access Token、SSH key，或 PyCharm 的 `Settings | Version Control | GitHub` 登录。

后续同步：

```bash
git status
git add README.md MIGRATION_LOG.md official_pytorch siba_jittor scripts tools docs demo logs results checkpoints data_manifests configs
git commit -m "Complete SIBA Jittor reproduction artifacts"
git push
```

不要上传数据集、第三方仓库、缓存和 2,824 张完整 PNG 副本。GitHub 保留代码、数据清单、完整日志、CSV、曲线、检查点和代表性可视化；全部输出可由公开脚本重新生成。

## 8. 现场演示顺序

1. PyCharm 展开 `official_pytorch/` 与 `siba_jittor/`，说明同路径对应关系。
2. 展示 `models/SIBA.py`、`base_blocks/restormer.py`、`loss/loss.py`、`train.py`。
3. 打开两套 Conda 解释器，分别打印框架版本。
4. 展示逐模块 Notebook 的已验证输出，不重新执行全部耗时测试。
5. 打开 60 轮日志和损失曲线。
6. 运行一次 TNO Jittor 推理，展示新输出、时间和图像。
7. 展示三数据集官方权重、自训练指标和 706 张输出对齐。
8. 打开 GitHub 页面，展示已经推送的同一仓库版本。
