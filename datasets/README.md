# Dataset Directory

Dataset images are not included in this repository. Download the original
datasets into the dataset-first source tree:

```text
datasets/
`-- source/
    |-- MSRS/
    |   |-- train/{ir,vi}/       # 1,083 pairs
    |   `-- test/{ir,vi}/        # 361 pairs
    |-- RoadScene/
    |   |-- cropinfrared/
    |   `-- crop_LR_visible/
    |-- M3FD_Fusion/{Ir,Vis}/    # 300 pairs
    `-- TNO/{ir,vi}/             # 45 pairs
```

Then run `python prepare_data.py`. It validates the files, selects the fixed
200-pair RoadScene subset, resizes M3FD, and creates the model-ready tree:

```text
datasets/
`-- SIBA/
    |-- train/
    |   |-- ir/                 # MSRS 1,083 + RoadScene 200
    |   `-- vi/                 # 1,283 matching filenames
    `-- test/
        |-- MSRS/{ir,vi}/       # 361 pairs
        |-- M3FD_2x/{ir,vi}/    # 300 half-resolution pairs
        `-- TNO/{ir,vi}/        # 45 pairs
```

The training directory is a prepared mixture because SIBA trains one model on
MSRS and RoadScene together. M3FD and TNO are test datasets and must not be
placed under `train/`. With this layout, `python train.py` and `python test.py`
require no path edits.

See [`../data/README.md`](../data/README.md) for download links, preparation,
provenance and citations.
