# Dataset Provenance

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
