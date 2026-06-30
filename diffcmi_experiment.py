"""
DiffCMI: Diffusion-based Cross-Modal Imputation for Robust Multimodal Sentiment Analysis
=========================================================================================

Core idea:
  1) CCMA  : align text/audio/visual into a shared latent space via contrastive learning
  2) CLDI  : Conditional Latent Diffusion Imputer - generate missing-modality latent
             representations by denoising, conditioned on available modalities
  3) Fusion: cross-modal transformer fuses (real + generated) latents -> sentiment

Datasets: CMU-MOSEI (main), CMU-MOSI (benchmark), CH-SIMS v2 (cross-lingual)

----------------------------------------------------------------------
HOW TO RUN  (on your GPU machine)
----------------------------------------------------------------------
1. Install:
     pip install torch numpy scikit-learn tqdm

2. Get aligned feature pickles (recommended: MMSA packaged features):
     # MOSI / MOSEI: https://github.com/thuiar/MMSA  ->  download `unaligned_*.pkl` / `aligned_*.pkl`
     # CH-SIMS v2 : https://github.com/thuiar/ch-sims-v2  -> Google Drive feature files
   Put them under:  ./data/MOSEI/aligned_50.pkl  (etc.)

3. Quick pipeline test with synthetic data (no download needed):
     python diffcmi_experiment.py --synthetic --dataset mosei --epochs 3

4. Real run:
     python diffcmi_experiment.py --dataset mosei --data_path ./data/MOSEI/aligned_50.pkl \
            --missing_rate 0.5 --missing_type random --epochs 50

5. Full suite for the paper:
     python diffcmi_experiment.py --run_all --data_root ./data
----------------------------------------------------------------------
"""

import os
import math
import pickle
import random
import argparse
from typing import Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 0. REPRODUCIBILITY
# ============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 1. DATA
# ============================================================
class MSADataset(Dataset):
    """
    Works with MMSA-format pickle:
       data[split] = {'text':(N,T,Dt), 'audio':(N,T,Da), 'vision':(N,T,Dv),
                      'regression_labels':(N,), 'id':[...]}
    Falls back to synthetic arrays when feature dicts are passed directly.
    """
    def __init__(self, feats: Dict, missing_rate: float = 0.0, missing_type: str = "random",
                 fixed_mask: bool = False, mask_seed: int = 1234):
        self.text = torch.as_tensor(feats["text"], dtype=torch.float32)
        self.audio = torch.as_tensor(feats["audio"], dtype=torch.float32)
        self.vision = torch.as_tensor(feats["vision"], dtype=torch.float32)
        self.labels = torch.as_tensor(feats["labels"], dtype=torch.float32)
        self.missing_rate = missing_rate
        self.missing_type = missing_type
        # When fixed_mask=True (validation/test), every sample gets a deterministic
        # missing pattern that is identical across models and across epochs. This makes
        # the comparison fair and the numbers reproducible -- essential for the paper.
        self.fixed_mask = fixed_mask
        self.mask_seed = mask_seed

    def __len__(self):
        return len(self.labels)

    def _make_mask(self, idx: int = None) -> torch.Tensor:
        mask = torch.ones(3)  # [text, audio, vision]; 1=available 0=missing
        if self.missing_rate <= 0 and self.missing_type == "random":
            return mask
        # deterministic per-sample RNG for eval; global RNG (None) for train-time aug
        rng = random.Random(self.mask_seed + idx) if (self.fixed_mask and idx is not None) else random
        if self.missing_type == "random":
            for i in range(3):
                if rng.random() < self.missing_rate:
                    mask[i] = 0.0
            if mask.sum() == 0:                       # keep at least one modality
                mask[rng.randint(0, 2)] = 1.0
        elif self.missing_type == "text":
            mask[0] = 0.0
        elif self.missing_type == "audio":
            mask[1] = 0.0
        elif self.missing_type == "vision":
            mask[2] = 0.0
        return mask

    def __getitem__(self, idx):
        return {
            "text": self.text[idx],
            "audio": self.audio[idx],
            "vision": self.vision[idx],
            "label": self.labels[idx],
            "mask": self._make_mask(idx),
        }


def load_pickle_split(path: str, split: str) -> Dict:
    """Load one split from an MMSA-format pickle and normalise key names."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    d = data[split]
    # regression label key varies across versions
    if "regression_labels" in d:
        labels = np.array(d["regression_labels"]).reshape(-1)
    elif "labels" in d:
        labels = np.array(d["labels"]).reshape(-1)
    else:
        raise KeyError("No label field found in pickle split.")
    return {
        "text": np.asarray(d["text"], dtype=np.float32),
        "audio": np.asarray(d["audio"], dtype=np.float32),
        "vision": np.asarray(d["vision"], dtype=np.float32),
        "labels": labels.astype(np.float32),
    }


def make_synthetic(n, seq, dt, da, dv):
    """Synthetic data so the pipeline can be tested without downloads."""
    # Inject weak label-correlated signal so metrics aren't pure chance
    labels = np.random.uniform(-3, 3, n).astype(np.float32)
    base = labels[:, None, None]
    return {
        "text": (np.random.randn(n, seq, dt) + 0.3 * base).astype(np.float32),
        "audio": (np.random.randn(n, seq, da) + 0.2 * base).astype(np.float32),
        "vision": (np.random.randn(n, seq, dv) + 0.2 * base).astype(np.float32),
        "labels": labels,
    }


# ============================================================
# 2. ENCODERS
# ============================================================
class SeqEncoder(nn.Module):
    """Generic sequence encoder (1D-conv + pooling) for any modality."""
    def __init__(self, in_dim, hid):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, hid, 3, padding=1)
        self.conv2 = nn.Conv1d(hid, hid, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hid, hid)
        self.ln = nn.LayerNorm(hid)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):              # x:(B,T,in_dim)
        x = x.transpose(1, 2)          # (B,in_dim,T)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)   # (B,hid)
        return self.drop(self.ln(self.fc(x)))


# ============================================================
# 3. CONTRASTIVE CROSS-MODAL ALIGNMENT (CCMA)
# ============================================================
class CCMA(nn.Module):
    def __init__(self, hid, proj=64):
        super().__init__()
        mk = lambda: nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, proj))
        self.pt, self.pa, self.pv = mk(), mk(), mk()
        self.log_tau = nn.Parameter(torch.tensor(math.log(0.07)))

    def project(self, ht, ha, hv):
        zt = F.normalize(self.pt(ht), dim=-1)
        za = F.normalize(self.pa(ha), dim=-1)
        zv = F.normalize(self.pv(hv), dim=-1)
        return zt, za, zv

    def _info_nce(self, zi, zj):
        tau = self.log_tau.exp().clamp(0.01, 1.0)
        logits = zi @ zj.t() / tau
        labels = torch.arange(zi.size(0), device=zi.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

    def loss(self, zt, za, zv, mask):
        losses = []
        for (zi, zj, a, b) in [(zt, za, 0, 1), (zt, zv, 0, 2), (za, zv, 1, 2)]:
            sel = (mask[:, a] * mask[:, b]).bool()
            if sel.sum() > 1:
                losses.append(self._info_nce(zi[sel], zj[sel]))
        return torch.stack(losses).mean() if losses else torch.zeros((), device=zt.device)


# ============================================================
# 4. CONDITIONAL LATENT DIFFUSION IMPUTER (CLDI)
# ============================================================
def timestep_embedding(t, dim):
    """Sinusoidal timestep embedding."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class DenoiseNet(nn.Module):
    """Predicts noise eps added to a latent, conditioned on (timestep, available-context)."""
    def __init__(self, hid, t_dim=64):
        super().__init__()
        self.t_mlp = nn.Sequential(nn.Linear(t_dim, hid), nn.SiLU(), nn.Linear(hid, hid))
        self.net = nn.Sequential(
            nn.Linear(hid * 2, hid * 2), nn.SiLU(),
            nn.Linear(hid * 2, hid * 2), nn.SiLU(),
            nn.Linear(hid * 2, hid),
        )
        self.t_dim = t_dim

    def forward(self, x_t, t, cond):           # x_t:(B,hid) cond:(B,hid)
        temb = self.t_mlp(timestep_embedding(t, self.t_dim))
        h = torch.cat([x_t + temb, cond], dim=-1)
        return self.net(h)                      # predicted noise


class CLDI(nn.Module):
    """
    Conditional latent DDPM operating in the hidden space.
    Trains one shared denoiser; modality identity is folded into `cond`.
    """
    def __init__(self, hid, T=100, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.T = T
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("acp", acp)             # alpha_bar
        self.register_buffer("sqrt_acp", acp.sqrt())
        self.register_buffer("sqrt_1m_acp", (1 - acp).sqrt())
        # modality embedding added to condition (text/audio/vision)
        self.mod_emb = nn.Embedding(3, hid)
        self.denoise = DenoiseNet(hid)

    def q_sample(self, x0, t, noise):
        return self.sqrt_acp[t][:, None] * x0 + self.sqrt_1m_acp[t][:, None] * noise

    def train_loss(self, x0, cond, mod_id):
        """Diffusion denoising loss for one modality batch."""
        B = x0.size(0)
        t = torch.randint(0, self.T, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        c = cond + self.mod_emb(torch.full((B,), mod_id, device=x0.device, dtype=torch.long))
        pred = self.denoise(x_t, t, c)
        return F.mse_loss(pred, noise)

    def _predict_x0(self, x_t, t, c):
        """Given x_t and predicted noise, recover the clean latent x0."""
        eps = self.denoise(x_t, t, c)
        return (x_t - self.sqrt_1m_acp[t][:, None] * eps) / self.sqrt_acp[t][:, None]

    def impute(self, cond, mod_id, steps=5):
        """
        Differentiable imputation used IDENTICALLY at train and test time.
        A short DDIM-style trajectory from noise to x0, conditioned on context.
        Kept differentiable (no @no_grad) so the imputed latent can be supervised
        by the downstream sentiment loss -- this is what makes generation actually
        serve the task and ties train/test distributions together.
        """
        B = cond.size(0)
        c = cond + self.mod_emb(torch.full((B,), mod_id, device=cond.device, dtype=torch.long))
        x = torch.randn(B, cond.size(1), device=cond.device)
        ts = torch.linspace(self.T - 1, 0, steps, device=cond.device).long()
        for ti in ts:
            t = torch.full((B,), int(ti), device=cond.device, dtype=torch.long)
            x = self._predict_x0(x, t, c)          # jump toward clean estimate
        return x

    @torch.no_grad()
    def sample(self, cond, mod_id):
        """Generate a latent for the missing modality via reverse diffusion."""
        B = cond.size(0)
        c = cond + self.mod_emb(torch.full((B,), mod_id, device=cond.device, dtype=torch.long))
        x = torch.randn(B, cond.size(1), device=cond.device)
        for ti in reversed(range(self.T)):
            t = torch.full((B,), ti, device=cond.device, dtype=torch.long)
            eps = self.denoise(x, t, c)
            alpha = 1 - self.betas[ti]
            acp = self.acp[ti]
            coef = (1 - alpha) / self.sqrt_1m_acp[ti]
            mean = (x - coef * eps) / alpha.sqrt()
            if ti > 0:
                x = mean + self.betas[ti].sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x


# ============================================================
# 5. FUSION + FULL MODEL
# ============================================================
class FusionTransformer(nn.Module):
    def __init__(self, hid, heads=4, layers=2):
        super().__init__()
        enc = nn.TransformerEncoderLayer(hid, heads, hid * 4, dropout=0.1, batch_first=True)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.cls = nn.Parameter(torch.randn(1, 1, hid))
        # Modality position embedding (text/audio/vision are distinct slots).
        self.mod_pos = nn.Parameter(torch.randn(1, 3, hid) * 0.02)
        # Availability embedding: index 0 = imputed/missing, 1 = real/available.
        # This is the key signal — it lets attention down-weight imputed slots and
        # trust real ones, instead of treating every modality identically.
        self.avail_emb = nn.Embedding(2, hid)

    def forward(self, ht, ha, hv, mask=None):
        B = ht.size(0)
        mods = torch.stack([ht, ha, hv], dim=1) + self.mod_pos      # (B,3,hid)
        if mask is not None:
            # mask: (B,3) with 1=available, 0=missing -> availability embedding
            mods = mods + self.avail_emb(mask.long())               # (B,3,hid)
        seq = torch.cat([self.cls.expand(B, -1, -1), mods], dim=1)  # (B,4,hid)
        return self.tr(seq)[:, 0]


class DiffCMI(nn.Module):
    def __init__(self, dims, hid=128, proj=64, T=100, out_dim=1):
        super().__init__()
        dt, da, dv = dims
        self.enc_t = SeqEncoder(dt, hid)
        self.enc_a = SeqEncoder(da, hid)
        self.enc_v = SeqEncoder(dv, hid)
        self.ccma = CCMA(hid, proj)
        self.cldi = CLDI(hid, T=T)
        self.fusion = FusionTransformer(hid)
        self.head = nn.Sequential(nn.Linear(hid, hid // 2), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(hid // 2, out_dim))

    def _aggregate(self, ht, ha, hv, mask):
        stk = torch.stack([ht, ha, hv], dim=1)            # (B,3,hid)
        m = mask.unsqueeze(-1)
        return (stk * m).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)

    def forward(self, text, audio, vision, mask, training=True):
        ht, ha, hv = self.enc_t(text), self.enc_a(audio), self.enc_v(vision)
        zt, za, zv = self.ccma.project(ht, ha, hv)
        align_loss = self.ccma.loss(zt, za, zv, mask)

        cond = self._aggregate(ht, ha, hv, mask)
        diff_loss = torch.zeros((), device=text.device)
        hf = [ht.clone(), ha.clone(), hv.clone()]
        encs = [ht, ha, hv]

        for mod_id in range(3):
            miss = (mask[:, mod_id] == 0)
            # Diffusion denoising objective: trained on AVAILABLE samples (target known).
            if training:
                avail = (mask[:, mod_id] == 1)
                if avail.sum() > 1:
                    diff_loss = diff_loss + self.cldi.train_loss(
                        encs[mod_id][avail], cond[avail], mod_id)
            # Imputation for the prediction path -- SAME mechanism in train & test,
            # so the fusion head always sees diffusion-generated latents (no
            # train/test distribution gap). The impute call is differentiable, so the
            # sentiment loss also shapes the generator toward task-useful latents.
            if miss.any():
                hf[mod_id][miss] = self.cldi.impute(cond[miss], mod_id)

        fused = self.fusion(hf[0], hf[1], hf[2], mask)
        pred = self.head(fused)
        return {"pred": pred, "align": align_loss, "diff": diff_loss / 3.0, "fused": fused}

    @torch.no_grad()
    def predict_with_uncertainty(self, text, audio, vision, mask, n_samples=10):
        """
        Generative uncertainty estimation -- the capability deterministic baselines
        lack. For each missing modality we draw n_samples independent diffusion
        imputations, producing a distribution of predictions per input. The mean is
        the point estimate; the std is an intrinsic, modality-aware uncertainty score.
        More missing / harder-to-impute inputs yield higher variance.
        Returns: mean_pred (B,), uncertainty (B,)  [both numpy-friendly tensors]
        """
        self.eval()
        ht, ha, hv = self.enc_t(text), self.enc_a(audio), self.enc_v(vision)
        cond = self._aggregate(ht, ha, hv, mask)
        preds = []
        for _ in range(n_samples):
            hf = [ht.clone(), ha.clone(), hv.clone()]
            for mod_id in range(3):
                miss = (mask[:, mod_id] == 0)
                if miss.any():
                    # each call re-samples noise -> a different plausible completion
                    hf[mod_id][miss] = self.cldi.impute(cond[miss], mod_id)
            fused = self.fusion(hf[0], hf[1], hf[2], mask)
            preds.append(self.head(fused).squeeze(-1))      # (B,)
        stacked = torch.stack(preds, dim=0)                  # (n_samples, B)
        mean_pred = stacked.mean(0)                          # (B,)
        uncertainty = stacked.std(0)                         # (B,) predictive std
        return mean_pred, uncertainty


# ============================================================
# 5b. BASELINE MODELS (for comparison tables)
# ============================================================
class ImputationBaseline(nn.Module):
    """
    Shared backbone for simple-imputation baselines.
    strategy in {'zero','mean'} decides how missing latents are filled.
    """
    def __init__(self, dims, hid=128, strategy="zero", out_dim=1):
        super().__init__()
        dt, da, dv = dims
        self.enc_t = SeqEncoder(dt, hid)
        self.enc_a = SeqEncoder(da, hid)
        self.enc_v = SeqEncoder(dv, hid)
        self.fusion = FusionTransformer(hid)
        self.head = nn.Sequential(nn.Linear(hid, hid // 2), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(hid // 2, out_dim))
        self.strategy = strategy
        # running mean buffers for 'mean' imputation
        self.register_buffer("mean_t", torch.zeros(hid))
        self.register_buffer("mean_a", torch.zeros(hid))
        self.register_buffer("mean_v", torch.zeros(hid))
        self.momentum = 0.99

    def _fill(self, h, miss, running_mean):
        if self.strategy == "zero":
            h[miss] = 0.0
        elif self.strategy == "mean":
            h[miss] = running_mean.detach()
        return h

    def forward(self, text, audio, vision, mask, training=True):
        ht, ha, hv = self.enc_t(text), self.enc_a(audio), self.enc_v(vision)
        if training:  # update running means on available samples
            with torch.no_grad():
                for h, buf, mi in [(ht, self.mean_t, 0), (ha, self.mean_a, 1), (hv, self.mean_v, 2)]:
                    av = (mask[:, mi] == 1)
                    if av.any():
                        buf.mul_(self.momentum).add_(h[av].mean(0) * (1 - self.momentum))
        hf = [ht.clone(), ha.clone(), hv.clone()]
        means = [self.mean_t, self.mean_a, self.mean_v]
        for mi in range(3):
            miss = (mask[:, mi] == 0)
            if miss.any():
                hf[mi] = self._fill(hf[mi], miss, means[mi])
        fused = self.fusion(hf[0], hf[1], hf[2], mask)
        pred = self.head(fused)
        zero = torch.zeros((), device=text.device)
        return {"pred": pred, "align": zero, "diff": zero, "fused": fused}


class MMINBaseline(nn.Module):
    """
    Simplified MMIN: a cascaded residual autoencoder reconstructs missing
    latents from available ones (deterministic, no diffusion / no alignment).
    Reference: Zhao et al., Missing Modality Imagination Network, ACL 2021.
    """
    def __init__(self, dims, hid=128, out_dim=1):
        super().__init__()
        dt, da, dv = dims
        self.enc_t = SeqEncoder(dt, hid)
        self.enc_a = SeqEncoder(da, hid)
        self.enc_v = SeqEncoder(dv, hid)
        # cascaded residual autoencoder: context -> missing latent
        self.imaginer = nn.ModuleList([
            nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, hid))
            for _ in range(3)
        ])
        self.fusion = FusionTransformer(hid)
        self.head = nn.Sequential(nn.Linear(hid, hid // 2), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(hid // 2, out_dim))

    def _aggregate(self, ht, ha, hv, mask):
        stk = torch.stack([ht, ha, hv], dim=1)
        return (stk * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)

    def forward(self, text, audio, vision, mask, training=True):
        ht, ha, hv = self.enc_t(text), self.enc_a(audio), self.enc_v(vision)
        cond = self._aggregate(ht, ha, hv, mask)
        encs = [ht, ha, hv]
        hf = [ht.clone(), ha.clone(), hv.clone()]
        recon = torch.zeros((), device=text.device)
        for mi in range(3):
            imagined = self.imaginer[mi](cond)
            miss = (mask[:, mi] == 0)
            if training:
                av = (mask[:, mi] == 1)
                if av.any():  # supervise imagination on available samples
                    recon = recon + F.mse_loss(self.imaginer[mi](cond[av]), encs[mi][av].detach())
            if miss.any():
                hf[mi][miss] = imagined[miss]
        fused = self.fusion(hf[0], hf[1], hf[2], mask)
        pred = self.head(fused)
        zero = torch.zeros((), device=text.device)
        # reuse 'diff' slot to carry the reconstruction loss
        return {"pred": pred, "align": zero, "diff": recon, "fused": fused}


def build_model(name, dims, args):
    """Factory: name in {diffcmi, zero, mean, mmin}."""
    if name == "diffcmi":
        return DiffCMI(dims, hid=args.hidden_dim, T=args.diffusion_steps)
    if name == "zero":
        return ImputationBaseline(dims, hid=args.hidden_dim, strategy="zero")
    if name == "mean":
        return ImputationBaseline(dims, hid=args.hidden_dim, strategy="mean")
    if name == "mmin":
        return MMINBaseline(dims, hid=args.hidden_dim)
    raise ValueError(f"unknown model {name}")


# ============================================================
# 6. METRICS
# ============================================================
def metrics(preds, labels, label_range=3):
    """
    label_range: 3 for MOSI/MOSEI ([-3,3]), 1 for SIMS ([-1,1]).
    Acc-2 uses sign agreement (non-negative vs negative), which is range-agnostic.
    The multi-class accuracy is reported over the dataset's own label grid.
    """
    from sklearn.metrics import accuracy_score, f1_score
    preds, labels = preds.reshape(-1), labels.reshape(-1)
    mae = float(np.mean(np.abs(preds - labels)))
    corr = float(np.corrcoef(preds, labels)[0, 1]) if np.std(preds) > 0 else 0.0
    bp, bl = (preds >= 0).astype(int), (labels >= 0).astype(int)
    acc2 = float(accuracy_score(bl, bp))
    f1 = float(f1_score(bl, bp, average="weighted"))
    r = label_range
    sp = np.clip(np.round(preds), -r, r).astype(int)
    sl = np.clip(np.round(labels), -r, r).astype(int)
    acc_multi = float(accuracy_score(sl, sp))
    return {"mae": mae, "corr": corr, "acc2": acc2, "acc7": acc_multi, "f1": f1}


# ============================================================
# 7. TRAIN / EVAL
# ============================================================
class Trainer:
    def __init__(self, model, lr=1e-4, w_align=0.1, w_diff=0.1, device=None, label_range=3):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, "min", patience=5, factor=0.5)
        self.crit = nn.MSELoss()
        self.w_align, self.w_diff = w_align, w_diff
        self.label_range = label_range

    def _step(self, batch):
        return (batch["text"].to(self.device), batch["audio"].to(self.device),
                batch["vision"].to(self.device), batch["label"].to(self.device),
                batch["mask"].to(self.device))

    def train_epoch(self, dl):
        self.model.train()
        tot = 0.0
        for b in dl:
            text, audio, vision, label, mask = self._step(b)
            out = self.model(text, audio, vision, mask, training=True)
            loss = (self.crit(out["pred"].squeeze(-1), label)
                    + self.w_align * out["align"] + self.w_diff * out["diff"])
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            tot += loss.item()
        return tot / len(dl)

    @torch.no_grad()
    def evaluate(self, dl):
        self.model.eval()
        P, L = [], []
        for b in dl:
            text, audio, vision, label, mask = self._step(b)
            out = self.model(text, audio, vision, mask, training=False)
            P.append(out["pred"].cpu().numpy())
            L.append(label.cpu().numpy())
        return metrics(np.concatenate(P), np.concatenate(L), label_range=self.label_range)

    def fit(self, tr, va, epochs=50, patience=15, ckpt="best.pt"):
        best, wait = 1e9, 0
        for e in range(epochs):
            loss = self.train_epoch(tr)
            m = self.evaluate(va)
            self.sched.step(m["mae"])
            print(f"Epoch {e+1:>3}/{epochs} | loss {loss:.4f} | "
                  f"val MAE {m['mae']:.4f} Acc2 {m['acc2']:.4f} F1 {m['f1']:.4f} Corr {m['corr']:.4f}")
            if m["mae"] < best:
                best, wait = m["mae"], 0
                torch.save(self.model.state_dict(), ckpt)
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stop at epoch {e+1}")
                    break
        self.model.load_state_dict(torch.load(ckpt))
        return self.model


# ============================================================
# 8. EXPERIMENT DRIVER
# ============================================================
# MMSA official feature dims: (text_dim, audio_dim, vision_dim), seq_len
# Text is BERT-768 for all (bert-base-uncased for MOSI/MOSEI, bert-base-chinese for SIMS).
# NOTE: for real data these are only used as fallback — actual dims are auto-inferred
#       from the loaded pickle, so a mismatch here can never crash a real run.
DATASET_DIMS = {
    "mosei":  ((768, 74, 35), 50),
    "mosi":   ((768, 5, 20), 50),
    "chsims": ((768, 33, 709), 39),
}
SYNTH_N = {"mosei": (1600, 200, 460), "mosi": (1284, 229, 686), "chsims": (2722, 647, 1034)}

# Real MMSA filenames differ per dataset (SIMS has no aligned version).
DATASET_FILES = {
    "mosei":  ["aligned_50.pkl", "unaligned_50.pkl"],
    "mosi":   ["aligned_50.pkl", "unaligned_50.pkl"],
    "chsims": ["aligned_39.pkl", "unaligned_39.pkl", "aligned_50.pkl"],
}


def infer_dims(feats):
    """Read (text_dim, audio_dim, vision_dim) straight from the data arrays."""
    return (feats["text"].shape[-1], feats["audio"].shape[-1], feats["vision"].shape[-1])


def build_loaders(args):
    dims, seq = DATASET_DIMS[args.dataset]
    if args.synthetic or not args.data_path:
        ntr, nva, nte = SYNTH_N[args.dataset]
        tr = make_synthetic(ntr, seq, *dims)
        va = make_synthetic(nva, seq, *dims)
        te = make_synthetic(nte, seq, *dims)
    else:
        tr = load_pickle_split(args.data_path, "train")
        va = load_pickle_split(args.data_path, "valid")
        te = load_pickle_split(args.data_path, "test")
        dims = infer_dims(tr)            # trust the real data, not the hardcoded table
        print(f"  [dims] inferred from data: text={dims[0]} audio={dims[1]} vision={dims[2]}",
              flush=True)
    # train: random masks each epoch (acts as augmentation, helps robustness)
    # val/test: FIXED deterministic masks so every model is judged on the exact
    #           same missing patterns -> fair, low-variance, reproducible numbers.
    mk = lambda f, sh, fixed: DataLoader(
        MSADataset(f, args.missing_rate, args.missing_type, fixed_mask=fixed),
        batch_size=args.batch_size, shuffle=sh, drop_last=sh)
    return mk(tr, True, False), mk(va, False, True), mk(te, False, True), dims


def _resolve_data(args, ds):
    """Auto-discover a dataset pickle under data_root; else fall back to synthetic."""
    args.dataset = ds
    args.data_path = None
    if args.data_root:
        for fname in DATASET_FILES[ds]:
            cand = os.path.join(args.data_root, ds.upper(), fname)
            if os.path.exists(cand):
                args.data_path = cand
                break
        args.synthetic = args.data_path is None
    return args


def evaluate_uncertainty(model, loader, label_range=3, n_samples=10, device=None):
    """
    Core experiment for the uncertainty story. Computes, on the test set:
      - spearman/pearson corr between predictive std and absolute error
        (does the model 'know when it's wrong'?)
      - selective-prediction curve: accuracy/MAE when we keep only the most
        confident X% of samples (rejecting high-uncertainty ones)
    Returns a dict ready to be saved to json and plotted later.
    """
    import numpy as np

    def _pearson(a, b):
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    def _spearman(a, b):
        # Spearman = Pearson on ranks; pure numpy, no scipy needed.
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        return _pearson(ra, rb)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    all_pred, all_unc, all_label = [], [], []
    for b in loader:
        text = b["text"].to(device); audio = b["audio"].to(device)
        vision = b["vision"].to(device); mask = b["mask"].to(device)
        mean_pred, unc = model.predict_with_uncertainty(text, audio, vision, mask, n_samples)
        all_pred.append(mean_pred.cpu().numpy())
        all_unc.append(unc.cpu().numpy())
        all_label.append(b["label"].numpy())
    pred = np.concatenate(all_pred); unc = np.concatenate(all_unc); lab = np.concatenate(all_label)
    abs_err = np.abs(pred - lab)

    # 1) Does uncertainty track error?
    sp = _spearman(unc, abs_err)
    pr = _pearson(unc, abs_err)

    # 2) Selective prediction: keep most-confident fraction, measure Acc2 + MAE
    order = np.argsort(unc)                       # ascending uncertainty
    curve = []
    for frac in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        k = max(1, int(len(order) * frac))
        idx = order[:k]
        p, l = pred[idx], lab[idx]
        acc2 = float(((p >= 0) == (l >= 0)).mean())
        mae = float(np.mean(np.abs(p - l)))
        curve.append({"coverage": frac, "acc2": acc2, "mae": mae})

    return {
        "unc_err_spearman": sp,
        "unc_err_pearson": pr,
        "mean_uncertainty": float(np.mean(unc)),
        "selective_curve": curve,
    }


def run_one(args, model_name="diffcmi"):
    set_seed(args.seed)
    tag = f"[{args.dataset}|{model_name}] miss={args.missing_type}@{args.missing_rate} synth={args.synthetic}"
    print("=" * 72 + f"\n{tag}\n" + "=" * 72, flush=True)
    tr, va, te, dims = build_loaders(args)
    model = build_model(model_name, dims, args)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    label_range = 1 if args.dataset == "chsims" else 3   # SIMS labels are in [-1,1]
    trainer = Trainer(model, lr=args.lr, label_range=label_range)
    ckpt = f"ckpt_{args.dataset}_{model_name}_{args.missing_type}{int(args.missing_rate*100)}.pt"
    trainer.fit(tr, va, epochs=args.epochs, ckpt=ckpt)
    res = trainer.evaluate(te)
    print(f"\nTEST [{model_name}] Acc7 {res['acc7']:.4f} | Acc2 {res['acc2']:.4f} | "
          f"F1 {res['f1']:.4f} | MAE {res['mae']:.4f} | Corr {res['corr']:.4f}", flush=True)

    # Uncertainty experiment -- only DiffCMI can do this (generative). Baselines can't.
    if model_name == "diffcmi" and getattr(args, "eval_uncertainty", False):
        try:
            model.load_state_dict(torch.load(ckpt))
            unc = evaluate_uncertainty(model, te, label_range=label_range,
                                       n_samples=getattr(args, "unc_samples", 10))
            print(f"[UNCERTAINTY] corr(unc, error): spearman={unc['unc_err_spearman']:.4f} "
                  f"pearson={unc['unc_err_pearson']:.4f} | mean_unc={unc['mean_uncertainty']:.4f}",
                  flush=True)
            print("[SELECTIVE PREDICTION] coverage -> Acc2 / MAE:")
            for c in unc["selective_curve"]:
                print(f"    {int(c['coverage']*100):3d}%  Acc2={c['acc2']:.4f}  MAE={c['mae']:.4f}",
                      flush=True)
            res["uncertainty"] = unc
        except Exception as e:
            print(f"[UNCERTAINTY] skipped due to: {e}", flush=True)
    return res


def save_json(obj, path):
    import json
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[saved] {path}", flush=True)


def run_all(args):
    """
    Full paper suite (all models + datasets + missing-rate sweep).
    Results streamed to results.json incrementally so a crash never loses progress.
    """
    models = ["zero", "mean", "mmin", "diffcmi"] if args.with_baselines else ["diffcmi"]
    out_path = os.path.join(args.out_dir, "results.json")
    os.makedirs(args.out_dir, exist_ok=True)
    summary = {}

    # --- Main comparison: three datasets, random missing 0.5 ---
    for ds in ["mosei", "mosi", "chsims"]:
        _resolve_data(args, ds)
        args.missing_type, args.missing_rate = "random", 0.5
        for m in models:
            key = f"{ds}_random50_{m}"
            summary[key] = run_one(args, m)
            save_json(summary, out_path)

    # --- Missing-rate sweep on MOSEI (all models) ---
    _resolve_data(args, "mosei")
    for r in [0.1, 0.3, 0.5, 0.7]:
        args.missing_type, args.missing_rate = "random", r
        for m in models:
            key = f"mosei_rate{int(r*100)}_{m}"
            summary[key] = run_one(args, m)
            save_json(summary, out_path)

    # --- Structured missing on MOSEI (our method) ---
    _resolve_data(args, "mosei")
    for mt in ["text", "audio", "vision"]:
        args.missing_type, args.missing_rate = mt, 1.0
        summary[f"mosei_{mt}miss_diffcmi"] = run_one(args, "diffcmi")
        save_json(summary, out_path)

    print("\n" + "=" * 80 + "\nFINAL SUMMARY\n" + "=" * 80, flush=True)
    for k, v in summary.items():
        print(f"{k:30s} Acc2={v['acc2']:.4f} F1={v['f1']:.4f} "
              f"MAE={v['mae']:.4f} Corr={v['corr']:.4f}", flush=True)
    save_json(summary, out_path)
    print(f"\n[DONE] all results in {out_path}", flush=True)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mosei", choices=["mosei", "mosi", "chsims"])
    p.add_argument("--model", default="diffcmi", choices=["diffcmi", "zero", "mean", "mmin"])
    p.add_argument("--data_path", default=None, help="path to MMSA-format .pkl")
    p.add_argument("--data_root", default=None, help="root dir for --run_all auto-discovery")
    p.add_argument("--synthetic", action="store_true", help="use synthetic data (no download)")
    p.add_argument("--missing_rate", type=float, default=0.5)
    p.add_argument("--missing_type", default="random", choices=["random", "text", "audio", "vision"])
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--diffusion_steps", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_all", action="store_true")
    p.add_argument("--with_baselines", action="store_true",
                   help="include zero/mean/MMIN baselines in --run_all")
    p.add_argument("--out_dir", default="./outputs", help="where results.json is written")
    p.add_argument("--eval_uncertainty", action="store_true",
                   help="run the generative uncertainty experiment (DiffCMI only)")
    p.add_argument("--unc_samples", type=int, default=10,
                   help="number of diffusion samples for uncertainty estimation")
    args = p.parse_args()
    if args.run_all:
        run_all(args)
    else:
        os.makedirs(args.out_dir, exist_ok=True)
        res = run_one(args, args.model)
        save_json({f"{args.dataset}_{args.missing_type}{int(args.missing_rate*100)}_{args.model}": res},
                  os.path.join(args.out_dir, f"result_{args.dataset}_{args.model}.json"))


if __name__ == "__main__":
    main()
