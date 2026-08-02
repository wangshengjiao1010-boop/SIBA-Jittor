# Dataset Provenance

## Storage contract

Datasets are not redistributed through this repository. The public code tracks only preparation logic and integrity manifests. Original downloads use the dataset-first layout under `datasets/source/`; `python prepare_data.py` generates the model-ready `datasets/SIBA` tree. Training and testing therefore require no machine-specific path edits.

```text
datasets/source/
|-- MSRS/{train,test}/{ir,vi}/
|-- RoadScene/{cropinfrared,crop_LR_visible}/
|-- M3FD_Fusion/{Ir,Vis}/
`-- TNO/{ir,vi}/
```

M3FD and TNO are evaluation datasets. They are intentionally not placed under
the training directory.

```text
datasets/
`-- SIBA/
    |-- train/{ir,vi}                 # 1,283 pairs
    `-- test/
        |-- MSRS/{ir,vi}              # 361 pairs
        |-- M3FD_2x/{ir,vi}           # 300 pairs
        `-- TNO/{ir,vi}               # 45 pairs
```

The prepared training directory combines 1,083 MSRS pairs and the fixed 200-pair RoadScene subset because the official SIBA protocol trains one model on both sources. The original downloaded datasets may be stored anywhere; `prepare_data.py` validates and materializes the exact training and test layout above.

## Training data

### MSRS

- Official SIBA link: <https://github.com/Linfeng-Tang/MSRS/tree/main/train>
- Frozen repository commit: `5fa66e33c5ba875eec4de4f5a5bf609bdcc566c5`
- Training pairs: `1083`
- Test pairs: `361`
- Pair sizes and SHA256 values are stored in `data/manifests/msrs_train.json` and `data/manifests/msrs_test.json`.

### RoadScene

- Official SIBA link: <https://github.com/hanna-xu/RoadScene>
- Frozen repository commit: `ab71420f8fc96396eeb97eecf3694909b3feb656`
- Available aligned pairs: `221`
- Modalities used: `cropinfrared` and `crop_LR_visible`
- Selected pairs: `200`
- Selection seed: `2025`
- Selected filenames and hashes are stored in `data/manifests/roadscene_200_seed2025.json`.

The paper and official repository state that 200 RoadScene pairs are selected randomly, but neither the filenames nor the random seed are released. The deterministic subset in this reproduction is shared by PyTorch and Jittor and is not claimed to be identical to the authors' undisclosed subset.

## Test data

### M3FD

- Official SIBA link: <https://github.com/JinyuanLiu-CV/TarDAL>
- Official Google Drive archive: `M3FD_Fusion.zip`
- Archive SHA256: `ec33d031bbd26697b75061972786526cdd815ee8111586813427d155ec522dfc`
- Pairs: `300`
- The archive contains `Ir/` and `Vis/` with identical filenames.
- The SIBA paper, Section 4.1, requires all methods on M3FD to be evaluated after resizing every image to half its original width and height. All 300 pairs are retained; this is the paper protocol, not a reduced-data experiment. The released `utils/resize_resolution.py` uses Pillow LANCZOS and integer floor division.
- The GitHub backup `CharlesShan-hub/M3FD-Fusion-Backup` was checked only as a mirror: all 600 image files were byte-identical to the official archive.
- Half-resolution pair hashes are stored in `data/manifests/m3fd_2x_test.json`.

### TNO

- Official SIBA Google Drive folder: <https://drive.google.com/drive/folders/1yURIsV9R9kEYLQovQ-vPogUkXqrIZswA>
- Infrared Google export SHA256: `9b6380ae5447c35d340550ad861709ecda2d925760df3d322dd25fc191c92c7f`
- Visible Google export SHA256: `d99cd760719a633e72973607a8084bbbf39ccf27c3058981a664ede14b296736`
- Pairs: `45`
- Filenames: `01.png` through `45.png`
- Pair hashes are stored in `data/manifests/tno_test.json`.

A commonly mirrored 25-pair TNO subset was initially inspected and rejected because it was incomplete relative to the official SIBA folder. All final experiments use the 45-pair official download.

## Evaluation code

- Official SIBA link: <https://github.com/Linfeng-Tang/Evaluation-for-Image-Fusion>
- Frozen repository commit: `f5f055bcadb49c22fb734c3498aef6c56fc71f2a`
- Reported metrics: VIF, SCD, MI, Qabf, SSIM, MS-SSIM, and FMI.

## Dataset citations

The following entries come from the corresponding official dataset or framework repositories. Users should also review each upstream license and citation notice.

```bibtex
@article{Tang2022PIAFusion,
  title={PIAFusion: A progressive infrared and visible image fusion network based on illumination aware},
  author={Tang, Linfeng and Yuan, Jiteng and Zhang, Hao and Jiang, Xingyu and Ma, Jiayi},
  journal={Information Fusion},
  year={2022},
  publisher={Elsevier}
}

@inproceedings{xu2020aaai,
  title={FusionDN: A Unified Densely Connected Network for Image Fusion},
  author={Xu, Han and Ma, Jiayi and Le, Zhuliang and Jiang, Junjun and Guo, Xiaojie},
  booktitle={Proceedings of the Thirty-Fourth AAAI Conference on Artificial Intelligence},
  year={2020}
}

@inproceedings{liu2022target,
  title={Target-aware Dual Adversarial Learning and a Multi-scenario Multi-Modality Benchmark to Fuse Infrared and Visible for Object Detection},
  author={Liu, Jinyuan and Fan, Xin and Huang, Zhanbo and Wu, Guanyao and Liu, Risheng and Zhong, Wei and Luo, Zhongxuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5802--5811},
  year={2022}
}

@article{toet2017tno,
  title={The TNO multiband image data collection},
  author={Toet, Alexander},
  journal={Data in Brief},
  volume={15},
  pages={249--251},
  year={2017}
}

@article{hu2020jittor,
  title={Jittor: a novel deep learning framework with meta-operators and unified graph execution},
  author={Hu, Shi-Min and Liang, Dun and Yang, Guo-Ye and Yang, Guo-Wei and Zhou, Wen-Yang},
  journal={Science China Information Sciences},
  volume={63},
  number={222103},
  pages={1--21},
  year={2020}
}
```
