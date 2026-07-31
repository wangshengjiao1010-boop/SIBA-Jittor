# SIBA 源码与 Jittor 迁移讲解

## 1. 阅读主线

SIBA 的完整执行链为：

1. `loader/train_loader.py` 或 `loader/test_loader.py` 读取配准的红外/可见光图像；
2. `models/SIBA.py` 构建双模态特征主干、四个 CBSM 查询分支和四路交叉注意力；
3. `base_blocks/` 提供 SE、SE-ResNet、Restormer 自注意力、交叉注意力和 GDFN；
4. `loss/loss.py` 计算 Laplacian、强度和 Sobel 三项监督；
5. `train.py` 完成 60 轮优化，`test.py` 完成亮度融合和 RGB 重建。

Jittor 版保留官方 13 个 Python 文件的相对路径。框架兼容逻辑集中在 `compat/`，对照实验脚本集中在 `tests/`，不混入网络定义。

## 2. 逐文件对照

### `args/args_SIBA.py`

- 官方职责：统一训练路径和超参数。
- 关键配置：60 epochs、batch 4、patch 128、初始学习率 `1e-4`、StepLR `25/0.5`、权重衰减 `0`。
- Jittor 迁移：文件保持一致；运行时路径参数由顶层脚本覆盖，不改变默认算法配置。

### `base_blocks/SE.py`

- 官方职责：全局平均池化后经两层全连接和 Sigmoid 生成通道权重。
- Jittor 迁移：`torch.nn` 替换为 `jittor.nn`，`forward` 改为 `execute`；池化、降维、升维和逐通道乘法顺序不变。

### `base_blocks/cbsm.py`

- 官方职责：将单通道源图映射到 48 通道查询特征，执行顺序为 `Conv-PReLU-Conv-PReLU-SE`。
- Jittor 迁移：仅替换模块 API 和入口名称；卷积核、步幅、padding、通道数及返回值不变。

### `base_blocks/se_resnet.py`

- 官方职责：两层 3x3 卷积提取特征，SE 重标定后与 1x1 残差支路相加。
- Jittor 迁移：卷积、激活、SE 和残差相加的顺序保持一致。

### `base_blocks/restormer.py`

- 官方职责：实现 `TransformerBlock_SA`、`TransformerBlock_CA`、自注意力、交叉注意力、GDFN 和 LayerNorm。
- 自注意力：Q/K/V 均来自当前模态潜在特征。
- 交叉注意力：Q 来自 CBSM 源图分支，K/V 来自另一模态潜在特征。
- Jittor 迁移：`einops.rearrange` 改用 `jittor.einops.rearrange`；`forward` 改为 `execute`；L2 normalize 按 PyTorch 的 `max(norm, 1e-12)` 规则显式实现。
- 参数完整性：两套实现均为 137 个状态张量、565,941 个可训练参数；注意力缩放参数和 LayerNorm 权重均被 Jittor 注册。

### `models/SIBA.py`

- 官方职责：定义完整 SIBA 拓扑。
- 前向顺序：保存两幅源图并生成负变换；提取红外/可见光潜在特征；生成四个 CBSM 查询；执行四路 I-SCA/V-SCA；按官方顺序拼接；经两级 SE-ResNet 和输出卷积生成单通道融合结果。
- Jittor 迁移：`torch.cat` 改为 `jt.concat`，`timm.trunc_normal_` 改为 `jt.init.trunc_normal_`，`forward` 改为 `execute`；模块数量、通道和调用顺序不变。

### `loader/train_loader.py`

- 官方职责：按文件名排序并断言红外/可见光配对，灰度读取，归一化到 `[0,1]`，在同一坐标裁剪 128x128 patch。
- Jittor 迁移：继承 Jittor `Dataset` 并通过 `set_attrs` 设置 batch 和长度。
- 对照实验扩展：可选读取固定的样本顺序与裁剪坐标；默认不传 `--schedule` 时仍使用官方随机裁剪和框架数据加载行为。

### `loader/test_loader.py`

- 官方职责：读取测试对，分解可见光 Y/Cb/Cr，返回文件名和原始尺寸。
- Jittor 迁移：替换张量与 transform API，并显式设置数据集长度；配对和色彩分解逻辑不变。

### `loss/loss.py`

- 官方职责：`JointGrad` 约束较强 Laplacian 响应，`Fusionloss` 返回强度损失和 Sobel 梯度损失。
- 总损失：`10 * L_laplacian + 0.1 * L_intensity + L_sobel`。
- Jittor 迁移：Jittor 没有 Kornia，因而按 Kornia 0.7.0 的归一化 3x3 Laplacian 核、reflect padding 和分组卷积直接复现；Sobel 核设为不求梯度。

### `utils/RGB2YCrBb.py`

- 官方职责：RGB 与 YCbCr 的正反变换及 `[0,1]` 截断。
- Jittor 迁移：矩阵与拼接操作改用 Jittor；测试阶段仍只融合 Y，Cb/Cr 按官方代码取源图均值后恢复 RGB。

### `utils/resize_resolution.py`

- 官方职责：将 M3FD 全部 300 对图像的宽和高各缩小一半。
- Jittor 迁移：文件不依赖深度学习框架，保持一致。
- 协议说明：这是 SIBA 论文第 4.1 节的测试协议，不是减少样本或缩水实验。

### `train.py`

- 官方职责：创建 SIBA、Adam、StepLR 和三项损失，执行反向传播、全局 L2 梯度裁剪、60 轮训练和权重保存。
- Jittor 迁移：使用 `optimizer.backward`；通过 `compat/pytorch_clip.py` 和 `compat/pytorch_adam.py` 复现 PyTorch 1.10 的操作顺序。
- 可追溯扩展：可选加载 PyTorch 导出的 NPZ 初始化、固定训练日程、逐 batch 四项损失 CSV、运行元数据和 SHA256；默认参数仍可直接执行官方训练协议。

### `test.py`

- 官方职责：加载权重，逐图融合 Y 通道，恢复 RGB，裁剪并保存结果。
- Jittor 迁移：加载 `.pkl` 权重，保存前将 CHW 转为 Jittor transform 所需的 HWC；文件名、尺寸恢复和色彩重建保持一致。

## 3. 新增兼容与验证代码

### `compat/pytorch_clip.py`

按 PyTorch 1.10 的逐梯度范数、全局 L2 范数、`1e-6` 稳定项和缩放顺序实现裁剪。共享参考梯度下最大误差为 `4.8894e-9`。

### `compat/pytorch_adam.py`

按 PyTorch 1.10 的一阶矩、二阶矩、偏置修正和 epsilon 位置实现 Adam。共享参考梯度下一步参数最大误差为 `2.9802e-8`。

### `tests/export_pytorch_alignment.py` / `tests/check_jittor_alignment.py`

PyTorch 导出固定输入、137 个参数张量、主要模块激活、三项损失、全部梯度、裁剪结果和一步更新；Jittor 读取同一参考逐项比较。

### `tests/export_shared_initialization.py`

在 PyTorch 中固定 seed `2025` 创建一次模型，同时保存 PyTorch checkpoint 和框架无关 NPZ。两框架加载 NPZ 后比较参数内容 SHA256，而不是只比较随机种子文字。

### `tests/generate_training_schedule.py`

为 1,283 对训练图像生成完整 60 轮样本顺序和裁剪坐标。该日程只用于受控框架对比，使数据顺序不再成为混杂变量。

### `scripts/run_shared_comparison_screen.sh`

在 `screen -S kk` 中依次完成数据清单校验、逐模块对齐、共享初始化导出、两框架 60 轮训练、四项损失绘图、三数据集完整推理和 GPU 监控。脚本检测到旧正式运行时拒绝覆盖。

## 4. 已验证对齐结果

| 检查项 | 结果 |
|---|---:|
| 主要激活最大绝对误差 | `2.0206e-4` |
| 复合损失最大绝对误差 | `2.9802e-6` |
| 原生梯度余弦相似度 | `0.999945` |
| 原生梯度相对 L2 | `1.0508%` |
| 一步更新相对 L2 | `6.7269%` |
| 共享参考梯度下 Adam 最大误差 | `2.9802e-8` |
| 作者权重下 706 对输出最大像素差 | `1/255` |

这些结果支持“推理迁移高度一致、训练功能完整、优化兼容实现正确”。原生梯度和一步更新没有达到逐元素严格等价，因此不表述为两个框架训练轨迹完全相同。
