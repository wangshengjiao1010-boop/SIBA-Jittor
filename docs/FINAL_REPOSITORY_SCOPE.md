# 最终仓库范围与归档规则

## 1. GitHub 正式保留

- `official_pytorch/`：冻结的官方 PyTorch 源码和作者权重。
- `siba_jittor/`：同路径 Jittor 迁移实现。
- `scripts/`：环境、数据、训练、推理和 screen 脚本。
- `tools/`：源码审计、数值对齐、指标、性能、曲线和完整性验证。
- `demo/`：成对逐模块 Notebook 和现场演示 Notebook。
- `data_manifests/`：完整文件名、尺寸和 SHA256 清单。
- `checkpoints/`：两套完整 60 轮最终权重。
- `logs/`：环境、对齐、60 轮训练、推理、评测和演示验证日志。
- `results/`：CSV、JSON、曲线、性能表和代表性可视化。
- `docs/`：选择审计、数据来源、迁移记录、逐文件审计、代码讲解和演示说明。
- `README.md`、`MIGRATION_LOG.md`、`configs/`。

## 2. 本地保留但不上传

- `datasets/`：完整训练与测试数据，按数据集许可单独下载。
- `third_party/`：Jittor-Sprouts、SFDFusion 范例和 MATLAB 评测仓库。
- `results/full_*/**/*.png`：全部 2,824 张融合图，体积过大，可由脚本重新生成。
- `results/official_checkpoint_alignment_*/**/*.png`：完整官方权重输出。
- `deliverables/`、`ppt_work/`、缓存和浏览器临时文件。
- `detele/`：被移出正式仓库的历史版本和本地调试备份。

## 3. 保留 smoke 与模块测试的原因

`tools/smoke_test_jittor.py`、`logs/smoke/` 和模块 Notebook 不是最终指标，但它们记录了环境、前向、反向和训练入口的调试过程。学习要求明确要求记录调试问题和实现过程，因此这些文件属于验证证据，不是无关代码。

`tools/demo_train_step.py` 和 20 步演示同样只用于现场证明训练入口可执行。README 和 Notebook 已明确说明它们不用于最终指标。

## 4. 归档对象

以下历史文件放入本地 `detele/`，不上传 GitHub：

- 旧版 PPT 生成器；
- 优秀视频抽帧分析脚本；
- smoke 训练临时 checkpoint；
- PyCharm、Playwright、Jittor 和 pip 缓存；
- PPT 中间导出目录。

归档不等于删除实验证据。正式 60 轮日志、最终 checkpoint、三数据集指标、逐图 CSV、对齐报告和代表性图像全部保留。

## 5. GitHub 大文件策略

最终仓库当前跟踪内容约几十 MB，单个正式 checkpoint 约 2.2 MB，不需要 Git LFS。完整数据集和全部融合 PNG 不进入 Git。README 提供数据准备、完整推理和评测命令，任何人可重新生成完整输出。
