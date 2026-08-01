# SIBA-Jittor 真实性与完整性审计

## 审计范围

- 论文：Wang et al., *The Source Image is the Best Attention for Infrared and Visible Image Fusion*, ICCV 2025。
- 官方代码：`Afreshbird/SIBA`，固定提交 `880a1ddf9eaa610c64e5f25f87fbb146448addc9`。
- 迁移对象：官方 13 个 Python 文件及 Jittor 对应文件。
- 实验对象：1,283 对训练图像、MSRS/M3FD/TNO 共 706 对测试图像、训练日志、权重、逐图指标和可视化。

## 完整性结论

| 项目 | 状态 | 证据 |
|---|---|---|
| 官方 Python 文件覆盖 | 13/13 | `tests/audit_source.py`，无缺失文件、类或函数 |
| 网络参数 | 137 个状态张量，565,941 参数 | PyTorch/Jittor 对齐导出 |
| 训练协议 | 60 epochs，batch 4，patch 128 | `args/args_SIBA.py`、训练元数据 |
| 训练数据 | 1,083 MSRS + 200 RoadScene | `data/manifests/combined_training_1283.json` |
| 测试数据 | 361 MSRS + 300 M3FD_2x + 45 TNO | 三份测试 manifest |
| 作者权重推理 | 两框架均完成 706 对 | `results/alignment/official_checkpoint/` |
| 逐图指标 | 7 项指标、每图 CSV | `results/metrics/` |

## 数据真实性

1. 数据 manifest 记录文件名、尺寸和 SHA256；红外/可见光文件名必须一一对应。
2. 706 对本地测试数据已按 manifest 逐文件复验。
3. M3FD 使用全部 300 对，只按论文第 4.1 节统一缩小宽高，不减少样本。
4. RoadScene 作者只公开“随机选择 200 对”，未公开名单和 seed。本复现从公开的 221 对配准图像中固定 seed `2025` 选择 200 对，并让两框架共用；不宣称与作者私有名单相同。

## 结果真实性

1. 未发现随机生成融合图、占位图、手工填写指标或伪造训练日志冒充正式结果的直接证据。
2. 作者权重下，PyTorch 与 Jittor 的 706 对输出逐文件比较，文件名和尺寸一致，全局最大像素差为一个 uint8 灰度级。
3. 指标由 SIBA 官方链接的 `Linfeng-Tang/Evaluation-for-Image-Fusion` 定义计算，保留 MATLAB 日志、逐图 CSV 和汇总 CSV。
4. MATLAB 只用于复现论文链接的评价指标；网络训练和推理均为 Python，Jittor 迁移代码不依赖 MATLAB。
5. 历史两套独立 60 轮训练证明完整训练可收敛，但因框架随机初始化和 shuffle 序列不同，不用于证明逐 batch 等价。
6. 受控实验由 PyTorch 导出一次初始化，并向两框架提供相同样本顺序和裁剪坐标；最终结论只在 `EXPERIMENT_COMPLETE`、完整元数据和指标均存在后更新。

## 性能报告边界

- 论文硬件：TITAN RTX 24 GB。
- 复现硬件：RTX 3090 24 GB。
- 两框架速度对比只使用同一 RTX 3090 上的 CUDA 同步计时。
- 论文 `test.py` 的未同步 CUDA 计时保留作协议记录，但不作为真实性能结论。
- 不能把 RTX 3090 运行时间写成论文同硬件复现结果。

## 不能宣称“一模一样”的原因

1. 作者未公开 RoadScene 200 对的具体文件名和随机种子。
2. 论文文字描述 RGB 到 YCbCr 训练，官方训练 loader 实际读取灰度图；迁移遵循公开代码。
3. 原生梯度相对 L2 约 `1.05%`，一步更新相对 L2 约 `6.73%`；方向高度一致，但不是逐元素相同。
4. 复现 GPU 型号与论文不同。

## 审计判定

- 功能覆盖：完整，未发现算法分支、损失项、训练轮数、测试集或输出流程删减。
- Jittor 推理迁移：高可信。
- Jittor 训练能力：已由完整 60 轮运行验证。
- 训练逐步严格数值等价：尚未证明，不作该表述。
- 论文级完全同条件复现：受未公开数据子集和硬件差异限制，不成立。
- 数据或结果造假：未发现直接证据；最终可追溯性依赖保留日志、manifest、checkpoint 与哈希。
