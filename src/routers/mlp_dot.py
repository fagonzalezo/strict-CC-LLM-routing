"""
mlp_dot.py — MLP dot-product router baseline.

Two EmbeddingRefiner encoders (query: R^{d_q} → R^{target_dim},
LLM: R^{d_l} → R^{target_dim}) map inputs to a shared L2-normalised
latent space.  The routing score is

    score(q, l) = sigmoid(s · z_q · z_l + b)

where s > 0 and b are learnable scalars trained end-to-end with
per-pair binary cross-entropy.

Encoder architecture matches the EmbedKit default refiner
(embedkit.improvement.model.EmbeddingRefiner):
    n_layers × [Linear → BatchNorm1d → ReLU → Dropout] + Linear → L2-normalize
"""

from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


# ── Internal helpers ──────────────────────────────────────────────────────────

def _bce(P: np.ndarray, S: np.ndarray, eps: float = 1e-7) -> float:
    p = np.clip(P, eps, 1.0 - eps)
    y = S.astype(np.float32)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def _top1_acc(P: np.ndarray, S: np.ndarray) -> float:
    best = np.argmax(P, axis=1)
    return float(S[np.arange(len(best)), best].mean())


def _pq_auc(P: np.ndarray, S: np.ndarray) -> float:
    aucs = []
    for i in range(S.shape[0]):
        y = S[i]
        if y.min() == y.max():
            continue
        aucs.append(roc_auc_score(y.astype(float), P[i]))
    return float(np.mean(aucs)) if aucs else float("nan")


# ── Network ───────────────────────────────────────────────────────────────────

class _MLPEncoder(nn.Module):
    """MLP projector head mirroring the EmbedKit EmbeddingRefiner architecture.

    n_layers × [Linear → BatchNorm1d → ReLU → Dropout] + Linear(→ target_dim)
    Output is optionally L2-normalised (normalize_output=True by default).
    """

    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        normalize_output: bool = True,
    ):
        super().__init__()
        self.normalize_output = normalize_output
        layers: list[nn.Module] = []
        in_d = input_dim
        for _ in range(n_layers):
            layers += [
                nn.Linear(in_d, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            in_d = hidden_dim
        layers.append(nn.Linear(in_d, target_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        if self.normalize_output:
            z = F.normalize(z, dim=1)
        return z


class _MLPDotModel(nn.Module):
    def __init__(
        self,
        query_dim: int,
        llm_dim: int,
        target_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float,
        normalize_output: bool,
        temperature_init: float,
        learn_temperature: bool,
    ):
        super().__init__()
        self.f_q = _MLPEncoder(
            input_dim=query_dim,
            target_dim=target_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            normalize_output=normalize_output,
        )
        self.f_l = _MLPEncoder(
            input_dim=llm_dim,
            target_dim=target_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            normalize_output=normalize_output,
        )
        # s = exp(log_s) is always positive; init so s = temperature_init
        self.log_s = nn.Parameter(
            torch.tensor(math.log(temperature_init), dtype=torch.float32),
            requires_grad=learn_temperature,
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def score(self, z_q: torch.Tensor, z_l: torch.Tensor) -> torch.Tensor:
        """sigmoid(s · z_q @ z_l.T + b)  →  (Nq, Nl) in (0, 1)."""
        return torch.sigmoid(torch.exp(self.log_s) * (z_q @ z_l.T) + self.bias)

    def forward(self, q: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        return self.score(self.f_q(q), self.f_l(l))


# ── Router ────────────────────────────────────────────────────────────────────

class MLPDotRouter:
    """MLP dot-product router.

    Encodes queries and LLMs into a shared L2-normalised latent space via
    separate MLP encoders (architecture = EmbedKit default refiner), then
    predicts routing scores as sigmoid(s · z_q · z_l + b).

    Sklearn-style contract identical to DMKDMRRouter:
        fit(train_q, train_l, S_train, val_q, val_l, S_val) → self
        predict(test_q, test_l) → (Nq_test, Nl_test) float32 in [0, 1]

    Parameters
    ----------
    target_dim : int
        Shared latent dimension (both encoders project to this).
    hidden_dim, n_layers, dropout, normalize_output
        EmbeddingRefiner architecture defaults — match EmbedKit defaults.
    temperature_init : float
        Initial value of the learnable scale s.  Required because L2-
        normalised dot products live in [-1, 1], squashing sigmoid output
        to ≈(0.27, 0.73) without scaling.
    learn_temperature : bool
        If False, s is frozen at temperature_init throughout training.
    lr, weight_decay : float
        AdamW hyperparameters.
    epochs, batch_size : int
        Training budget.  Each step encodes the query batch and all train
        LLMs, then computes BCE over the full (batch, Nl_train) score matrix.
    seed : int
        Controls torch/numpy RNG for reproducibility.
    device : str | None
        Torch device; None → CUDA if available, else CPU.
    verbose : bool
        Print training progress every 20 epochs.
    val_objective : str
        Criterion for best-epoch selection within each fit.
        ``"bce"`` (default, backward-compatible): argmin val BCE.
        ``"pq_auc"``: argmax val pq_auc; tiebreak argmin val BCE.
        Exposes ``val_pq_auc_`` when set to ``"pq_auc"``.
    """

    def __init__(
        self,
        target_dim: int = 64,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        normalize_output: bool = True,
        temperature_init: float = 10.0,
        learn_temperature: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 200,
        batch_size: int = 256,
        seed: int = 0,
        device: str | None = None,
        verbose: bool = True,
        val_objective: str = "bce",
    ) -> None:
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.normalize_output = normalize_output
        self.temperature_init = temperature_init
        self.learn_temperature = learn_temperature
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.device = device
        self.verbose = verbose
        self.val_objective = val_objective

        # set by fit()
        self.val_bce_: float | None = None
        self.val_top1_: float | None = None
        self.val_pq_auc_: float | None = None
        self.best_epoch_: int | None = None
        self._model: _MLPDotModel | None = None
        self._device: torch.device | None = None

    def fit(
        self,
        train_q_emb: np.ndarray,   # (Nq_train, d_q)  float32
        train_l_emb: np.ndarray,   # (Nl_train, d_l)  float32
        S_train: np.ndarray,       # (Nq_train, Nl_train) {0,1}
        val_q_emb: np.ndarray,     # (Nq_val, d_q)
        val_l_emb: np.ndarray,     # (Nl_val, d_l)
        S_val: np.ndarray,         # (Nq_val, Nl_val) {0,1}
    ) -> "MLPDotRouter":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        device = torch.device(
            self.device if self.device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._device = device

        train_q  = torch.from_numpy(train_q_emb.astype(np.float32)).to(device)
        train_l  = torch.from_numpy(train_l_emb.astype(np.float32)).to(device)
        S_train_t = torch.from_numpy(S_train.astype(np.float32)).to(device)

        val_q_t  = torch.from_numpy(val_q_emb.astype(np.float32)).to(device)
        val_l_t  = torch.from_numpy(val_l_emb.astype(np.float32)).to(device)

        model = _MLPDotModel(
            query_dim=train_q.shape[1],
            llm_dim=train_l.shape[1],
            target_dim=self.target_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
            normalize_output=self.normalize_output,
            temperature_init=self.temperature_init,
            learn_temperature=self.learn_temperature,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        n_q = train_q.shape[0]
        best_val_bce    = float("inf")
        best_val_pq_auc = -1.0
        best_state: dict | None = None

        if self.verbose:
            print(
                f"[MLPDotRouter] fit  device={device}  "
                f"train={n_q}q×{train_l.shape[0]}l  "
                f"val={val_q_t.shape[0]}q×{val_l_t.shape[0]}l  "
                f"target_dim={self.target_dim}  epochs={self.epochs}"
            )

        t0 = time.time()
        for epoch in range(1, self.epochs + 1):
            model.train()
            perm = torch.randperm(n_q, device=device)

            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_q, self.batch_size):
                idx      = perm[start : start + self.batch_size]
                q_batch  = train_q[idx]       # (B, d_q)
                s_batch  = S_train_t[idx]     # (B, Nl_train)

                optimizer.zero_grad()
                z_q    = model.f_q(q_batch)   # (B, target_dim)
                z_l    = model.f_l(train_l)   # (Nl_train, target_dim)
                scores = model.score(z_q, z_l)  # (B, Nl_train)
                loss   = F.binary_cross_entropy(scores, s_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches  += 1

            # ── Val evaluation ────────────────────────────────────────────
            model.eval()
            with torch.no_grad():
                z_q_val = model.f_q(val_q_t)
                z_l_val = model.f_l(val_l_t)
                P_val   = model.score(z_q_val, z_l_val).cpu().numpy()

            val_bce  = _bce(P_val, S_val)
            val_top1 = _top1_acc(P_val, S_val)

            if self.val_objective == "pq_auc":
                val_pq = _pq_auc(P_val, S_val)
                pq_cmp = val_pq if not math.isnan(val_pq) else -float("inf")
                if (pq_cmp > best_val_pq_auc) or (
                    pq_cmp == best_val_pq_auc and val_bce < best_val_bce
                ):
                    best_val_pq_auc = pq_cmp
                    best_val_bce    = val_bce
                    best_state      = copy.deepcopy(model.state_dict())
                    self.best_epoch_ = epoch
            else:
                if val_bce < best_val_bce:
                    best_val_bce = val_bce
                    best_state   = copy.deepcopy(model.state_dict())
                    self.best_epoch_ = epoch

            if self.verbose and (epoch == 1 or epoch % 20 == 0 or epoch == self.epochs):
                print(
                    f"  epoch {epoch:3d}/{self.epochs}  "
                    f"train_bce={epoch_loss/n_batches:.4f}  "
                    f"val_bce={val_bce:.4f}  val_top1={val_top1:.4f}  "
                    f"s={model.log_s.exp().item():.2f}  "
                    f"[{time.time()-t0:.0f}s]"
                )

        # ── Restore best checkpoint ───────────────────────────────────────
        assert best_state is not None
        model.load_state_dict(best_state)
        model.eval()
        self._model = model

        with torch.no_grad():
            z_q_val = model.f_q(val_q_t)
            z_l_val = model.f_l(val_l_t)
            P_val   = model.score(z_q_val, z_l_val).cpu().numpy()
        self.val_bce_    = _bce(P_val, S_val)
        self.val_top1_   = _top1_acc(P_val, S_val)
        if self.val_objective == "pq_auc":
            self.val_pq_auc_ = _pq_auc(P_val, S_val)

        if self.verbose:
            print(
                f"[MLPDotRouter] best_epoch={self.best_epoch_}  "
                f"val_bce={self.val_bce_:.5f}  val_top1={self.val_top1_:.4f}  "
                f"total={time.time()-t0:.1f}s"
            )
        return self

    def predict(
        self,
        test_q_emb: np.ndarray,   # (Nq_test, d_q)
        test_l_emb: np.ndarray,   # (Nl_test, d_l)
    ) -> np.ndarray:              # (Nq_test, Nl_test) float32 in [0, 1]
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        q = torch.from_numpy(test_q_emb.astype(np.float32)).to(self._device)
        l = torch.from_numpy(test_l_emb.astype(np.float32)).to(self._device)
        self._model.eval()
        with torch.no_grad():
            z_q = self._model.f_q(q)
            z_l = self._model.f_l(l)
            P   = self._model.score(z_q, z_l).cpu().numpy()
        return P.astype(np.float32)
