#!/usr/bin/env python3
import json
from pathlib import Path


def markdown(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


def write_notebook(path, cells, display_name):
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": display_name,
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.8"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path)


def build_live_demo(project_root):
    output = project_root / "demo" / "SIBA_Jittor_现场演示.ipynb"
    cells = [
        markdown(
            """
# SIBA-Jittor 现场演示

本 Notebook 只组织真实代码、真实日志和真实输出，不生成替代数据。完整 60 轮训练与 706 对测试结果已经保存在仓库中；现场训练只运行少量真实训练步，用于展示代码可执行性，不作为论文指标。
"""
        ),
        markdown("## 1. 环境与源码完整性"),
        code(
            """
import json
from pathlib import Path

PROJECT_ROOT = Path('/root/autodl-tmp/SIBA-Jittor')
print((PROJECT_ROOT / 'logs/environment/reproduction_environment.txt').read_text())
source_audit = json.loads((PROJECT_ROOT / 'docs/source_audit_final_20260728.json').read_text())
print(json.dumps(source_audit['comparison'], ensure_ascii=False, indent=2))
"""
        ),
        markdown("## 2. 数值对齐的分层结论"),
        code(
            """
assessment = json.loads((PROJECT_ROOT / 'logs/alignment/alignment_assessment_final_20260728.json').read_text())
assessment['measured'], assessment['conclusions']
"""
        ),
        markdown("## 3. 真实训练图像上的短训练演示"),
        code(
            """
!cd "{PROJECT_ROOT}" && bash scripts/start_demo_training_screen.sh
print('在终端执行 screen -r kk，可连续查看每一步真实损失。')
"""
        ),
        code(
            """
shared_initial = PROJECT_ROOT / 'logs/demo_shared_initial/SIBA_seed2025_initial.json'
print(shared_initial.read_text())
"""
        ),
        code(
            """
latest_demo = Path((PROJECT_ROOT / 'logs/latest_demo.txt').read_text().strip())
print((latest_demo / 'train.log').read_text())
"""
        ),
        markdown("## 4. 完整 60 轮训练日志与损失曲线"),
        code(
            """
from IPython.display import Image, display

display(Image(filename=str(PROJECT_ROOT / 'results/training_analysis_20260727_siba_official_protocol/loss_curve.png')))
print((PROJECT_ROOT / 'results/training_analysis_20260727_siba_official_protocol/training_log_summary.json').read_text())
"""
        ),
        markdown("## 5. Jittor 推理"),
        code(
            """
JITTOR_CHECKPOINT = PROJECT_ROOT / 'checkpoints/jittor_msrs_roadscene_60e_20260727_siba_official_protocol/07-27-04-52/SIBA_epoch60.pkl'
DEMO_OUTPUT = PROJECT_ROOT / 'results/demo_jittor_tno'
JITTOR_PYTHON = Path('/root/autodl-tmp/envs/siba_jittor/bin/python')
!"{JITTOR_PYTHON}" "{PROJECT_ROOT / 'tools/run_inference.py'}" --framework jittor --checkpoint "{JITTOR_CHECKPOINT}" --data-dir /root/autodl-tmp/datasets/SIBA/test/TNO --output "{DEMO_OUTPUT}" --use-cuda --warmup-runs 3 --timing-mode synchronized
"""
        ),
        markdown("## 6. 真实融合结果与指标"),
        code(
            """
from PIL import Image as PILImage
from IPython.display import display

images = sorted(DEMO_OUTPUT.glob('*.png')) + sorted(DEMO_OUTPUT.glob('*.jpg'))
print('新生成融合图数量:', len(images))
for path in images[:3]:
    print(path.name)
    display(PILImage.open(path))
"""
        ),
        code(
            """
import pandas as pd

metrics = pd.read_csv(PROJECT_ROOT / 'results/metrics_20260727_siba_official_protocol/metrics_summary.csv')
metrics[metrics['dataset'] == 'TNO']
"""
        ),
        markdown("## 7. PyTorch 与 Jittor 输出一致性"),
        code(
            """
for dataset in ['MSRS', 'M3FD_2x', 'TNO']:
    report = json.loads((PROJECT_ROOT / f'results/output_alignment_20260727_siba_official_protocol/{dataset}/summary.json').read_text())
    print(dataset, report)
"""
        ),
    ]
    write_notebook(output, cells, "SIBA Jittor")


def build_alignment_demo(project_root):
    output = project_root / "demo" / "SIBA_PyTorch_Jittor_对齐演示.ipynb"
    cells = [
        markdown(
            """
# SIBA PyTorch / Jittor 对齐演示

本 Notebook 只读取项目中已经保存的真实源码审计、数值对齐、训练、推理和指标文件。它不生成替代实验数据，也不修改论文算法。
"""
        ),
        code(
            """
import json
from pathlib import Path

PROJECT_ROOT = Path('/root/autodl-tmp/SIBA-Jittor')
RUN_TAG = '20260727_siba_official_protocol'
"""
        ),
        markdown("## 1. 官方文件与 Jittor 文件对应关系"),
        code(
            """
source_audit = json.loads((PROJECT_ROOT / 'docs/source_audit_final_20260728.json').read_text())
source_audit['comparison']
"""
        ),
        code(
            """
official_code = (PROJECT_ROOT / 'official_pytorch/models/SIBA.py').read_text()
jittor_code = (PROJECT_ROOT / 'siba_jittor/models/SIBA.py').read_text()
print('PyTorch SIBA.py lines:', len(official_code.splitlines()))
print('Jittor  SIBA.py lines:', len(jittor_code.splitlines()))
"""
        ),
        markdown("## 2. 数据读取与配对检查"),
        code(
            """
print(json.dumps(json.loads((PROJECT_ROOT / 'logs/alignment/data_loader_report.json').read_text()), ensure_ascii=False, indent=2))
print(json.dumps(json.loads((PROJECT_ROOT / 'logs/alignment/training_dataset_validation.json').read_text()), ensure_ascii=False, indent=2))
"""
        ),
        markdown("## 3. 前向、损失、梯度与一步更新"),
        code(
            """
alignment = json.loads((PROJECT_ROOT / 'logs/alignment/alignment_assessment_final_20260728.json').read_text())
alignment['measured'], alignment['conclusions'], alignment['interpretation']
"""
        ),
        markdown("## 4. 官方权重下的 706 张输出对齐"),
        code(
            """
for dataset in ['MSRS', 'M3FD_2x', 'TNO']:
    path = PROJECT_ROOT / f'results/output_alignment_{RUN_TAG}/{dataset}/summary.json'
    print(dataset)
    print(json.dumps(json.loads(path.read_text()), ensure_ascii=False, indent=2))
"""
        ),
        markdown("## 5. 两框架完整 60 轮训练"),
        code(
            """
from IPython.display import Image, display

curve = PROJECT_ROOT / f'results/training_analysis_{RUN_TAG}/loss_curve.png'
display(Image(filename=str(curve)))
print((PROJECT_ROOT / f'results/training_analysis_{RUN_TAG}/training_log_summary.json').read_text())
"""
        ),
        markdown("## 6. 指标、速度与显存"),
        code(
            """
import pandas as pd

display(pd.read_csv(PROJECT_ROOT / f'results/metrics_{RUN_TAG}/metrics_summary.csv'))
display(pd.read_csv(PROJECT_ROOT / f'results/performance_summary_{RUN_TAG}/inference_timing.csv'))
print((PROJECT_ROOT / f'results/performance_summary_{RUN_TAG}/gpu_monitor_summary.json').read_text())
"""
        ),
        markdown("## 7. 复现边界"),
        code(
            """
print('推理迁移：官方权重下高度一致。')
print('训练功能：完成全量 60 轮并收敛。')
print('严格训练步等价：未通过，不能宣称逐步完全相同。')
print('RoadScene 训练子集：作者未公开具体 200 对名单，不能宣称完全相同。')
"""
        ),
    ]
    write_notebook(output, cells, "SIBA Alignment")


def build_pytorch_module_demo(project_root):
    output = project_root / "demo" / "SIBA_PyTorch_逐模块测试.ipynb"
    cells = [
        markdown(
            """
# SIBA PyTorch 逐模块测试

本 Notebook 使用官方 PyTorch 源码和官方权重生成基准。受控张量只用于数值单元测试，不作为训练数据或论文实验结果。
"""
        ),
        markdown("## 1. 环境与路径"),
        code(
            """
import json
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import Image, display

PROJECT_ROOT = Path('/root/autodl-tmp/SIBA-Jittor')
TORCH_PYTHON = Path('/root/autodl-tmp/envs/siba_torch/bin/python')
OFFICIAL_CHECKPOINT = PROJECT_ROOT / 'official_pytorch/checkpoint/SIBA_epoch60.pth'
TEST_ROOT = PROJECT_ROOT / 'logs/demo_module_tests'
PYTORCH_REFERENCE = TEST_ROOT / 'pytorch_seed2025.npz'
TEST_ROOT.mkdir(parents=True, exist_ok=True)
print('project:', PROJECT_ROOT)
print('checkpoint:', OFFICIAL_CHECKPOINT)
"""
        ),
        code(
            """
!"{TORCH_PYTHON}" -c "import torch; print('PyTorch', torch.__version__); print('CUDA', torch.cuda.is_available())"
"""
        ),
        markdown("## 2. 官方源码与数据检查"),
        code(
            """
source_audit = json.loads((PROJECT_ROOT / 'docs/source_audit_final_20260728.json').read_text())
print(json.dumps(source_audit['comparison'], ensure_ascii=False, indent=2))
print((PROJECT_ROOT / 'logs/alignment/training_dataset_validation.json').read_text())
print((PROJECT_ROOT / 'logs/alignment/test_dataset_pairing.json').read_text())
"""
        ),
        markdown("## 3. 生成PyTorch基准"),
        code(
            """
!"{TORCH_PYTHON}" "{PROJECT_ROOT / 'tools/export_pytorch_alignment.py'}" --project-root "{PROJECT_ROOT}" --checkpoint "{OFFICIAL_CHECKPOINT}" --output "{PYTORCH_REFERENCE}" --batch-size 1 --height 32 --width 32 --seed 2025 --device cuda
"""
        ),
        code(
            """
reference = np.load(PYTORCH_REFERENCE)
metadata = json.loads(PYTORCH_REFERENCE.with_suffix('.json').read_text())
print(metadata)
print('导出数组数量:', len(reference.files))

def show_arrays(names):
    rows = []
    for name in names:
        value = np.asarray(reference[name])
        rows.append({
            'name': name,
            'shape': str(value.shape),
            'min': float(value.min()),
            'max': float(value.max()),
            'mean': float(value.mean()),
            'std': float(value.std()),
        })
    return pd.DataFrame(rows)
"""
        ),
        markdown("## 4. 固定输入与权重"),
        code(
            """
display(show_arrays(['input_ir', 'input_vi']))
parameter_keys = [name for name in reference.files if name.startswith('parameter_initial__')]
print('参数张量数量:', len(parameter_keys))
print('示例参数:', parameter_keys[:5])
"""
        ),
        markdown("## 5. SE与Res-SE特征提取"),
        code(
            """
display(show_arrays(['activation__ir_conv', 'activation__vi_conv']))
"""
        ),
        markdown("## 6. LayerNorm、归一化与自注意力"),
        code(
            """
display(show_arrays(['activation__ir_sa_0', 'activation__vi_sa_0']))
"""
        ),
        markdown("## 7. CBSM源图权重"),
        code(
            """
display(show_arrays([
    'activation__weight_ir', 'activation__weight_irI',
    'activation__weight_vi', 'activation__weight_viI',
]))
"""
        ),
        markdown("## 8. 四路交叉注意力"),
        code(
            """
display(show_arrays([
    'activation__ir2vi_ca_0', 'activation__irI2vi_ca_0',
    'activation__vi2ir_ca_0', 'activation__viI2ir_ca_0',
]))
"""
        ),
        markdown("## 9. 特征拼接与输出"),
        code(
            """
display(show_arrays(['activation__mixed', 'activation__fuse_conv', 'activation__output']))
"""
        ),
        markdown("## 10. Laplacian、Intensity与Sobel损失"),
        code(
            """
loss_names = ['loss_laplacian', 'loss_intensity', 'loss_sobel', 'loss_total']
display(show_arrays(loss_names))
print('Loss = 10 × Laplacian + 0.1 × Intensity + Sobel')
"""
        ),
        markdown("## 11. 反向传播"),
        code(
            """
gradient_keys = [name for name in reference.files if name.startswith('gradient_preclip__')]
gradient_l2 = np.sqrt(sum(float((reference[name].astype(np.float64) ** 2).sum()) for name in gradient_keys))
print('梯度张量数量:', len(gradient_keys))
print('裁剪前全局L2范数:', gradient_l2)
"""
        ),
        markdown("## 12. 梯度裁剪"),
        code(
            """
postclip_keys = [name for name in reference.files if name.startswith('gradient_postclip__')]
postclip_l2 = np.sqrt(sum(float((reference[name].astype(np.float64) ** 2).sum()) for name in postclip_keys))
print('PyTorch返回的裁剪前范数:', float(reference['clip_total_norm']))
print('裁剪后全局L2范数:', postclip_l2)
"""
        ),
        markdown("## 13. Adam一步更新"),
        code(
            """
update_keys = [name for name in reference.files if name.startswith('parameter_update__')]
update_l2 = np.sqrt(sum(float((reference[name].astype(np.float64) ** 2).sum()) for name in update_keys))
print('参数更新张量数量:', len(update_keys))
print('一步更新全局L2范数:', update_l2)
"""
        ),
        markdown("## 14. PyTorch基准说明"),
        code(
            """
print('本Notebook输出是Jittor逐模块测试的参考值。')
print('最终实验指标仍来自完整60轮训练和706对测试，不来自32×32受控张量。')
"""
        ),
    ]
    write_notebook(output, cells, "SIBA PyTorch Tests")


def build_jittor_module_demo(project_root):
    output = project_root / "demo" / "SIBA_Jittor_逐模块测试.ipynb"
    cells = [
        markdown(
            """
# SIBA Jittor 逐模块测试

本 Notebook 读取PyTorch生成的同一输入、同一权重和同一步训练基准，再逐项检查Jittor结果。测试结论按实际误差输出，不把功能通过写成严格数值相等。
"""
        ),
        markdown("## 1. 环境与路径"),
        code(
            """
import json
from pathlib import Path

import pandas as pd
from IPython.display import Image, display

PROJECT_ROOT = Path('/root/autodl-tmp/SIBA-Jittor')
JITTOR_PYTHON = Path('/root/autodl-tmp/envs/siba_jittor/bin/python')
OFFICIAL_CHECKPOINT = PROJECT_ROOT / 'official_pytorch/checkpoint/SIBA_epoch60.pth'
TEST_ROOT = PROJECT_ROOT / 'logs/demo_module_tests'
PYTORCH_REFERENCE = TEST_ROOT / 'pytorch_seed2025.npz'
JITTOR_REPORT = TEST_ROOT / 'jittor_seed2025.json'
TEST_ROOT.mkdir(parents=True, exist_ok=True)
"""
        ),
        code(
            """
!"{JITTOR_PYTHON}" -c "import jittor as jt; jt.flags.use_cuda=1; print('Jittor', jt.__version__); print('CUDA', jt.flags.use_cuda)"
"""
        ),
        markdown("## 2. 运行全部Jittor检查"),
        code(
            """
if not PYTORCH_REFERENCE.exists():
    raise FileNotFoundError('请先运行 SIBA_PyTorch_逐模块测试.ipynb 生成PyTorch基准。')
!"{JITTOR_PYTHON}" "{PROJECT_ROOT / 'tools/check_jittor_alignment.py'}" --project-root "{PROJECT_ROOT}" --checkpoint "{OFFICIAL_CHECKPOINT}" --reference "{PYTORCH_REFERENCE}" --output "{JITTOR_REPORT}" --use-cuda
"""
        ),
        code(
            """
report = json.loads(JITTOR_REPORT.read_text())
checks = report['checks']

def show_checks(names):
    rows = []
    for name in names:
        rows.append({'name': name, **checks[name]})
    return pd.DataFrame(rows)

def show_prefix(prefix):
    return show_checks([name for name in checks if name.startswith(prefix)])

print('检查项数量:', len(checks))
print('Jittor版本:', report['jittor_version'])
"""
        ),
        markdown("## 3. 数据读取与配对"),
        code(
            """
print((PROJECT_ROOT / 'logs/alignment/data_loader_report.json').read_text())
print((PROJECT_ROOT / 'logs/alignment/training_dataset_validation.json').read_text())
print((PROJECT_ROOT / 'logs/alignment/test_dataset_pairing.json').read_text())
"""
        ),
        markdown("## 4. 权重键与初始参数"),
        code(
            """
print('checkpoint参数键:', report['checkpoint_parameter_keys'])
print('Jittor模型参数键:', report['model_parameter_keys'])
initial = show_prefix('parameter_initial__')
display(initial.sort_values('max_abs', ascending=False).head(10))
print('全部初始参数最大绝对误差:', initial['max_abs'].max())
"""
        ),
        markdown("## 5. SE与Res-SE特征提取"),
        code(
            """
display(show_checks(['activation__ir_conv', 'activation__vi_conv']))
"""
        ),
        markdown("## 6. LayerNorm、归一化与自注意力"),
        code(
            """
display(show_checks(['activation__ir_sa_0', 'activation__vi_sa_0']))
"""
        ),
        markdown("## 7. CBSM源图权重"),
        code(
            """
display(show_checks([
    'activation__weight_ir', 'activation__weight_irI',
    'activation__weight_vi', 'activation__weight_viI',
]))
"""
        ),
        markdown("## 8. 四路交叉注意力"),
        code(
            """
display(show_checks([
    'activation__ir2vi_ca_0', 'activation__irI2vi_ca_0',
    'activation__vi2ir_ca_0', 'activation__viI2ir_ca_0',
]))
"""
        ),
        markdown("## 9. 特征拼接与最终输出"),
        code(
            """
display(show_checks([
    'activation__mixed', 'activation__fuse_conv',
    'activation__output', 'manual_vs_model_output',
]))
"""
        ),
        markdown("## 10. Laplacian、Intensity与Sobel损失"),
        code(
            """
display(show_checks(['loss_laplacian', 'loss_intensity', 'loss_sobel', 'loss_total']))
"""
        ),
        markdown("## 11. 原生Jittor反向传播"),
        code(
            """
display(pd.DataFrame([{'stage': 'gradient_preclip', **report['aggregates']['gradient_preclip']}]))
print('相对L2误差约1.05%，不满足严格1e-3阈值。')
"""
        ),
        markdown("## 12. PyTorch语义梯度裁剪"),
        code(
            """
display(show_checks(['clip_total_norm']))
display(pd.DataFrame([
    {'stage': 'native_gradient_postclip', **report['aggregates']['gradient_postclip']},
    {'stage': 'reference_gradient_postclip', **report['aggregates']['reference_gradient_postclip']},
]))
"""
        ),
        markdown("## 13. PyTorch兼容Adam"),
        code(
            """
display(pd.DataFrame([
    {'stage': 'native_parameter_update', **report['aggregates']['parameter_update']},
    {'stage': 'reference_gradient_parameter_after_step', **report['aggregates']['reference_gradient_parameter_after_step']},
]))
print('相同参考梯度下，裁剪和Adam实现通过；原生一步更新不宣称严格相等。')
"""
        ),
        markdown("## 14. 分层测试结论"),
        code(
            """
display(pd.DataFrame([report['summary']]).T.rename(columns={0: 'value'}))
"""
        ),
        markdown("## 15. 官方权重下706张输出对齐"),
        code(
            """
RUN_TAG = '20260727_siba_official_protocol'
rows = []
for dataset in ['MSRS', 'M3FD_2x', 'TNO']:
    result = json.loads((PROJECT_ROOT / f'results/output_alignment_{RUN_TAG}/{dataset}/summary.json').read_text())
    rows.append({
        'dataset': dataset,
        'images': result['compared_images'],
        'max_abs_uint8': result['global_max_abs_uint8'],
        'mean_abs_uint8': result['global_mean_abs_uint8'],
        'filenames_shapes_match': result['all_filenames_and_shapes_match'],
    })
display(pd.DataFrame(rows))
"""
        ),
        markdown("## 16. 完整60轮训练"),
        code(
            """
display(Image(filename=str(PROJECT_ROOT / f'results/training_analysis_{RUN_TAG}/loss_curve.png')))
print((PROJECT_ROOT / f'results/training_analysis_{RUN_TAG}/training_log_summary.json').read_text())
"""
        ),
        markdown("## 17. 指标与性能"),
        code(
            """
display(pd.read_csv(PROJECT_ROOT / f'results/metrics_{RUN_TAG}/metrics_summary.csv'))
display(pd.read_csv(PROJECT_ROOT / f'results/performance_summary_{RUN_TAG}/inference_timing.csv'))
print((PROJECT_ROOT / f'results/performance_summary_{RUN_TAG}/gpu_monitor_summary.json').read_text())
"""
        ),
        markdown("## 18. 复现边界"),
        code(
            """
print('推理：官方权重下高度一致。')
print('训练：Jittor完成全量60轮并收敛。')
print('严格训练步等价：未通过，不能表述为逐步完全相同。')
print('RoadScene 200对名单：作者未公开，不能表述为训练集完全相同。')
"""
        ),
    ]
    write_notebook(output, cells, "SIBA Jittor Tests")


def main():
    project_root = Path(__file__).resolve().parents[1]
    build_live_demo(project_root)
    build_alignment_demo(project_root)
    build_pytorch_module_demo(project_root)
    build_jittor_module_demo(project_root)


if __name__ == "__main__":
    main()
