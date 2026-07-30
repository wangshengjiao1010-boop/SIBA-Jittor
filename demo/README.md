# SIBA-Jittor 现场演示

现场演示以 PyCharm 远程 SSH 为主，成对 Notebook 用于逐模块结果展示。PyTorch 先生成基准，Jittor 使用相同输入与权重逐项比较。完整实验结果来自已经完成的60轮训练和706对测试；32×32受控张量及20步短训练只用于代码验证。

正式环境名称为 `PytorchDome` 和 `JittorDome`。PyCharm 配置见 `docs/PYCHARM_REMOTE_GUIDE.md`。

## Notebook

1. `SIBA_PyTorch_逐模块测试.ipynb`
   - 使用官方PyTorch源码和官方权重生成707项基准。
   - 展示输入、初始参数、Res-SE、自注意力、CBSM、交叉注意力、输出、三项损失、梯度裁剪和Adam一步更新。
2. `SIBA_Jittor_逐模块测试.ipynb`
   - 读取完全相同的PyTorch输入和基准。
   - 逐模块显示最大绝对误差、平均绝对误差、相对误差和余弦相似度。
   - 展示原生梯度与一步更新的实际误差，不宣称严格训练步等价。
3. `SIBA_Jittor_现场演示.ipynb`
   - 运行20步真实训练并生成新权重。
   - 运行TNO推理并显示新生成的融合图。
4. `SIBA_PyTorch_Jittor_对齐演示.ipynb`
   - 汇总源码、完整训练、706张输出、指标和性能结果。

## 现场顺序

1. 打开两本逐模块测试Notebook，在左侧Outline展示完整测试目录。
2. 运行PyTorch基准单元，再运行Jittor逐模块检查。
3. 依次展示Res-SE、自注意力、CBSM、交叉注意力、损失、反向传播、裁剪和Adam。
4. 运行：

   ```bash
   bash scripts/demo.sh
   screen -r kk
   ```

   脚本会先由PyTorch导出固定随机种子2025的初始权重，再由Jittor加载同一权重训练20步。结果保存在 `logs/demo_<时间>/`。
5. 展示完整60轮训练曲线和日志。
6. 运行TNO推理，打开新生成图像、`summary.json` 和 `timing.csv`。
7. 展示706张官方权重输出对齐、三数据集指标和性能结果。
8. 最后打开公开GitHub仓库，不在视频中输入密码或token。

## 说明

- PyTorch/Jittor逐模块测试采用受控输入，作用是定位迁移误差，不是论文实验数据。
- 20步训练使用真实训练图像，但只证明训练代码可执行，不用于最终指标。
- 完整结论来自1,283对训练数据、60轮训练和706对测试。
- 官方权重推理高度一致；原生Jittor训练一步接近但没有达到严格数值相等。
