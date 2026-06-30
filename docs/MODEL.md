# Model Architecture

This document describes the components of DiffCMI as implemented in
`diffcmi_experiment.py`. See the paper for the full formulation.

## Pipeline

```
inputs ── encoders ── contrastive alignment ── context c
                                                   │
                                  ┌────────────────┴───────────────┐
                                  │ (genuine modalities, skip path)│
                                  ▼                                ▼
                          latent diffusion imputer ──► availability-aware fusion ──► prediction + uncertainty
```

## Components

### 1. Sequence encoders (`SeqEncoder`)
Each modality sequence is mapped to a fixed-length hidden vector of width `hid`
(default 128). The three encoders are independent.

### 2. Contrastive Cross-Modal Alignment (`CCMA`)
Projects the available modality embeddings into a shared space and applies an
InfoNCE objective over available modality pairs, so that the conditioning context
for the generator is semantically coherent across modalities. Pairs involving a
missing modality are masked out.

### 3. Conditional Latent Diffusion Imputer (`CLDI`)
A denoising diffusion model operating in the `hid`-dimensional latent space.

- **Training**: the noise-prediction objective is evaluated only on *available*
  modalities (where the clean target is known).
- **Inference / imputation**: `impute()` runs a short, differentiable
  deterministic-trajectory sampler. The same routine is used at train and test time,
  so the fusion module always consumes diffusion-generated latents — there is no
  train/test distribution gap.
- **Uncertainty**: `predict_with_uncertainty()` draws `N` independent imputations
  (default `N=10`), producing a predictive distribution. The mean is the point
  estimate; the standard deviation is the uncertainty score.

### 4. Availability-Aware Fusion (`FusionTransformer`)
A shallow transformer with a learnable classification token. Each modality slot
receives a modality-position embedding plus an **availability embedding** (one of two
learned vectors for "genuine" vs. "imputed"), so attention can down-weight imputed
features and trust genuine ones.

### 5. Baselines
For fair comparison, the baselines share DiffCMI's encoder and fusion backbone and
differ only in the imputation mechanism:
- `ImputationBaseline` — zero or running-mean fill.
- `MMINBaseline` — an MMIN-style deterministic autoencoder imputer.

All three (and DiffCMI) use the same availability-aware fusion, so differences isolate
the imputation mechanism.

## Evaluation protocol

The validation and test splits use a **fixed, deterministic missing pattern** (seeded
per sample), so every model is judged on identical missing patterns and the numbers are
reproducible. Training-time masks remain random and act as augmentation. This is
implemented via the `fixed_mask` flag in `MSADataset`.

## Uncertainty evaluation

`evaluate_uncertainty()` computes, on the test set:
- the Spearman / Pearson correlation between predictive std and absolute error, and
- the selective-prediction curve (accuracy and MAE as a function of coverage when the
  most uncertain inputs are deferred).

Correlation is computed with a pure-NumPy implementation, so SciPy is not required.
