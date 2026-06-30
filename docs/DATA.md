# Data Preparation

DiffCMI uses pre-extracted, word-aligned multimodal features. We do **not** redistribute
the datasets; please obtain them from the official sources below and place the `.pkl`
files under a data root directory.

## Datasets

| Dataset | Source | File used | Dims (text/audio/vision) | Label range |
|---|---|---|---|---|
| CMU-MOSI | [CMU-MultimodalSDK](https://github.com/A2Zadeh/CMU-MultimodalSDK) | `MOSI/aligned_50.pkl` | 768 / 5 / 20 | [-3, 3] |
| CMU-MOSEI | [CMU-MultimodalSDK](https://github.com/A2Zadeh/CMU-MultimodalSDK) | `MOSEI/aligned_50.pkl` | 768 / 74 / 35 | [-3, 3] |
| CH-SIMS v2 | [MMSA / CH-SIMS](https://github.com/thuiar/MMSA) | `CHSIMS/unaligned_39.pkl` | 768 / 25 / 177 | [-1, 1] |

Text features are BERT embeddings in all three datasets. The MMSA toolkit provides a
convenient packaged version of these `.pkl` files.

## Expected directory layout

```
data/
├── MOSI/
│   └── aligned_50.pkl
├── MOSEI/
│   └── aligned_50.pkl
└── CHSIMS/
    └── unaligned_39.pkl
```

Point the code at this directory:

- single dataset: `--data_path ./data/MOSI/aligned_50.pkl`
- full matrix: `--data_root ./data` (the loader auto-discovers each dataset)

## Pickle format

Each `.pkl` is a dict with `train` / `valid` / `test` splits, each containing
`text`, `audio`, `vision` feature arrays and a `labels` array. The loader in
`diffcmi_experiment.py` (`MSADataset`) reads these fields directly. If your packaged
features use slightly different key names, adjust the loading logic in `_resolve_data`.

## Notes

- **CH-SIMS v2** here refers to the supervised v2(s) version; its feature dimensions
  differ from the original CH-SIMS, so report v2 numbers when comparing.
- If a dataset file is missing, the code can fall back to **synthetic data**
  (`--synthetic`) so you can verify the pipeline end-to-end before downloading the
  real features.
