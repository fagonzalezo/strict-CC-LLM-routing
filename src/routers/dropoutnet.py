"""
dropoutnet.py — DropoutNet-style cold-start router.

Two-stage model, adapted from Volkovs, Yu, Poutanen, "DropoutNet: Addressing
Cold Start in Recommender Systems" (NeurIPS 2017); see experiments/router-eval/dropoutnet-clcrec-baselines/README.md:

Stage A (pretrain): fit a logistic matrix-factorization model on S_train over
Q_train x M_train only, giving preference embeddings U_q (per train query) and
V_m (per train LLM). Real binary labels are used directly (this project has
ground truth, not implicit feedback), so plain logistic MF replaces the
original paper's WMF pretraining.

Stage B (deep towers with input dropout): two MLP towers
    z_q = f_Q([U_q_or_zero, x_q]),  z_m = f_M([V_m_or_zero, y_m])
score(q, m) = sigmoid(s * z_q . z_m + b). During training, independently zero
each example's U row and/or V row (Bernoulli, p_drop_q / p_drop_m) — this
synthesizes the four-cell taxonomy (warm / cold-query / cold-model / cold-cold)
from S_train alone, before any held-out pool is touched.

At inference (Q_val/M_val/Q_test/M_test), there is no pretrained U/V row for
any entity in those pools by construction of the disjoint-pool problem, so
predict() always uses the all-zero branch on both sides — no mean-pool
fallback (the original paper's Eq. 4 alternative), since these splits
contribute zero training interactions and there is no adaptation period.

Loss is BCE against real S_ij throughout (both MF pretrain and deep towers),
not the original paper's L2 regression toward the pretrained score — avoids
compounding pretraining error into the deep-tower target.

Encoder architecture and training-loop shape mirror src/routers/mlp_dot.py
(MLPDotRouter): n_layers x [Linear -> BatchNorm1d -> ReLU -> Dropout] +
Linear -> L2-normalize, full-batch-vs-full-train-pool BCE per step (already
exhaustive negatives).
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


# ── Internal helpers (duplicated from mlp_dot.py — each router file in this
# codebase is self-contained; no shared base module exists between routers) ──

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

    n_layers x [Linear -> BatchNorm1d -> ReLU -> Dropout] + Linear(-> target_dim)
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


class _DropoutNetModel(nn.Module):
    """Deep towers over concat[pretrained-MF-embedding-or-zero, content].

    No separate Phi_Q/Phi_M content-projection network: per the design doc,
    those are "likely closer to identity/normalization" than a learned
    transform, so the concatenation is fed straight into f_q/f_m and their
    first Linear layer absorbs any needed projection.
    """

    def __init__(
        self,
        query_dim: int,
        llm_dim: int,
        mf_dim: int,
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
            input_dim=mf_dim + query_dim,
            target_dim=target_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            normalize_output=normalize_output,
        )
        self.f_m = _MLPEncoder(
            input_dim=mf_dim + llm_dim,
            target_dim=target_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            normalize_output=normalize_output,
        )
        self.log_s = nn.Parameter(
            torch.tensor(math.log(temperature_init), dtype=torch.float32),
            requires_grad=learn_temperature,
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def score(self, z_q: torch.Tensor, z_m: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(torch.exp(self.log_s) * (z_q @ z_m.T) + self.bias)

    def forward(
        self,
        U_in: torch.Tensor, x_q: torch.Tensor,
        V_in: torch.Tensor, y_m: torch.Tensor,
    ) -> torch.Tensor:
        z_q = self.f_q(torch.cat([U_in, x_q], dim=1))
        z_m = self.f_m(torch.cat([V_in, y_m], dim=1))
        return self.score(z_q, z_m)


# ── Stage A: logistic MF pretrain ───────────────────────────────────────────

def _fit_logistic_mf(
    S_train: np.ndarray,
    mf_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
):
    """Full-batch logistic matrix factorization on S_train (real binary labels).

    Q_train x M_train is small enough (tens of millions of entries at most,
    at this project's pool sizes) for a single dense matmul per epoch.

    No validation-based early stopping: Q_val/M_val have no U/V rows by
    construction of the disjoint-pool problem, so there is nothing to
    validate against. Regularization is weight_decay + a fixed epoch budget
    only — this is an explicit limitation, not an oversight (see class
    docstring / self.mf_train_bce_).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    Nq, Nl = S_train.shape
    U = nn.Parameter(0.01 * torch.randn(Nq, mf_dim, generator=g).to(device))
    V = nn.Parameter(0.01 * torch.randn(Nl, mf_dim, generator=g).to(device))
    bias = nn.Parameter(torch.zeros(1, device=device))
    opt = torch.optim.AdamW([U, V, bias], lr=lr, weight_decay=weight_decay)

    S_t = torch.from_numpy(S_train.astype(np.float32)).to(device)
    train_bce = float("nan")
    for _ in range(epochs):
        opt.zero_grad()
        logits = U @ V.T + bias
        loss = F.binary_cross_entropy_with_logits(logits, S_t)
        loss.backward()
        opt.step()
        train_bce = float(loss.item())

    return U.detach(), V.detach(), train_bce


# ── Router ────────────────────────────────────────────────────────────────────

class DropoutNetRouter:
    """DropoutNet-style cold-start router.

    Sklearn-style contract identical to MLPDotRouter/DMKDMRRouter:
        fit(train_q, train_l, S_train, val_q, val_l, S_val) -> self
        predict(test_q, test_l) -> (Nq_test, Nl_test) float32 in [0, 1]

    Invariant: val checkpointing and predict() ALWAYS use the all-zero
    CF-embedding branch on both sides. There is no mean-pool fallback
    (rejected by the design doc: Q_val/M_val/Q_test/M_test contribute zero
    training interactions by construction, and there is no adaptation
    period), so there is no branch-selection ambiguity to resolve.

    Parameters
    ----------
    mf_dim : int
        Dimensionality of the pretrained MF embeddings U_q / V_m.
    mf_epochs, mf_lr, mf_weight_decay : Stage-A (logistic MF pretrain)
        hyperparameters. mf_epochs is a FIXED budget — MF pretraining has no
        validation signal (see _fit_logistic_mf docstring); final train BCE
        is exposed as self.mf_train_bce_.
    p_drop_q, p_drop_m : float
        Per-step Bernoulli drop probability for the batch's U rows / for all
        Nl_train V rows respectively. Independent draws each step generalize
        the design doc's "per training example" zeroing to this codebase's
        batch-vs-full-pool training loop shape: every step's (B, Nl_train)
        score matrix already mixes all four cells (warm / cold-query /
        cold-model / cold-cold) as a byproduct of the two independent masks.
    target_dim, hidden_dim, n_layers, dropout, normalize_output,
    temperature_init, learn_temperature, lr, weight_decay, epochs,
    batch_size, seed, device, verbose, val_objective
        Same meaning as MLPDotRouter.
    """

    def __init__(
        self,
        mf_dim: int = 32,
        mf_epochs: int = 100,
        mf_lr: float = 1e-2,
        mf_weight_decay: float = 1e-4,
        p_drop_q: float = 0.5,
        p_drop_m: float = 0.5,
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
        self.mf_dim = mf_dim
        self.mf_epochs = mf_epochs
        self.mf_lr = mf_lr
        self.mf_weight_decay = mf_weight_decay
        self.p_drop_q = p_drop_q
        self.p_drop_m = p_drop_m
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
        self.mf_train_bce_: float | None = None
        self.val_bce_: float | None = None
        self.val_top1_: float | None = None
        self.val_pq_auc_: float | None = None
        self.best_epoch_: int | None = None
        self.cell_diag_: dict | None = None
        self.zero_branch_exposure_: dict | None = None
        self._model: _DropoutNetModel | None = None
        self._device: torch.device | None = None

    def fit(
        self,
        train_q_emb: np.ndarray,   # (Nq_train, d_q)  float32
        train_l_emb: np.ndarray,   # (Nl_train, d_l)  float32
        S_train: np.ndarray,       # (Nq_train, Nl_train) {0,1}
        val_q_emb: np.ndarray,     # (Nq_val, d_q)
        val_l_emb: np.ndarray,     # (Nl_val, d_l)
        S_val: np.ndarray,         # (Nq_val, Nl_val) {0,1}
    ) -> "DropoutNetRouter":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        device = torch.device(
            self.device if self.device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._device = device

        train_q = torch.from_numpy(train_q_emb.astype(np.float32)).to(device)
        train_l = torch.from_numpy(train_l_emb.astype(np.float32)).to(device)
        S_train_t = torch.from_numpy(S_train.astype(np.float32)).to(device)

        val_q_t = torch.from_numpy(val_q_emb.astype(np.float32)).to(device)
        val_l_t = torch.from_numpy(val_l_emb.astype(np.float32)).to(device)

        # ── Stage A: logistic MF pretrain on S_train ────────────────────
        if self.verbose:
            print(
                f"[DropoutNetRouter] Stage A: logistic MF pretrain  "
                f"mf_dim={self.mf_dim}  mf_epochs={self.mf_epochs}"
            )
        U, V, mf_train_bce = _fit_logistic_mf(
            S_train, self.mf_dim, self.mf_epochs, self.mf_lr,
            self.mf_weight_decay, device, self.seed,
        )
        self.mf_train_bce_ = mf_train_bce

        # ── Stage B: deep towers with input dropout ─────────────────────
        model = _DropoutNetModel(
            query_dim=train_q.shape[1],
            llm_dim=train_l.shape[1],
            mf_dim=self.mf_dim,
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
        n_l = train_l.shape[0]
        best_val_bce = float("inf")
        best_val_pq_auc = -1.0
        best_state: dict | None = None

        zero_val_U = torch.zeros(val_q_t.shape[0], self.mf_dim, device=device)
        zero_val_V = torch.zeros(val_l_t.shape[0], self.mf_dim, device=device)

        # Running counters for the realized (as opposed to configured) zero-branch
        # exposure fraction — a sanity check that BatchNorm's running stats, which
        # the always-zero val/predict branch depends on, actually saw enough
        # zero-branch mass during training.
        zero_q_count = zero_q_total = 0
        zero_m_count = zero_m_total = 0

        if self.verbose:
            print(
                f"[DropoutNetRouter] Stage B: fit  device={device}  "
                f"train={n_q}q x {n_l}l  val={val_q_t.shape[0]}q x {val_l_t.shape[0]}l  "
                f"target_dim={self.target_dim}  epochs={self.epochs}  "
                f"p_drop_q={self.p_drop_q}  p_drop_m={self.p_drop_m}"
            )

        t0 = time.time()
        for epoch in range(1, self.epochs + 1):
            model.train()
            perm = torch.randperm(n_q, device=device)

            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_q, self.batch_size):
                idx = perm[start : start + self.batch_size]
                q_batch = train_q[idx]        # (B, d_q)
                U_batch = U[idx]               # (B, mf_dim)
                s_batch = S_train_t[idx]       # (B, Nl_train)

                # Independent per-step Bernoulli row masks: one over the
                # batch's U rows, one over all Nl_train V rows. Every step's
                # score matrix already mixes all four cells as a byproduct.
                keep_q = (torch.rand(len(idx), 1, device=device) > self.p_drop_q).float()
                keep_m = (torch.rand(n_l, 1, device=device) > self.p_drop_m).float()
                U_in = U_batch * keep_q
                V_in = V * keep_m

                zero_q_total += len(idx)
                zero_q_count += int((keep_q == 0).sum().item())
                zero_m_total += n_l
                zero_m_count += int((keep_m == 0).sum().item())

                optimizer.zero_grad()
                scores = model(U_in, q_batch, V_in, train_l)  # (B, Nl_train)
                loss = F.binary_cross_entropy(scores, s_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # ── Val evaluation (always the zero/zero branch) ────────────
            model.eval()
            with torch.no_grad():
                P_val = model(zero_val_U, val_q_t, zero_val_V, val_l_t).cpu().numpy()

            val_bce = _bce(P_val, S_val)
            val_top1 = _top1_acc(P_val, S_val)

            if self.val_objective == "pq_auc":
                val_pq = _pq_auc(P_val, S_val)
                pq_cmp = val_pq if not math.isnan(val_pq) else -float("inf")
                if (pq_cmp > best_val_pq_auc) or (
                    pq_cmp == best_val_pq_auc and val_bce < best_val_bce
                ):
                    best_val_pq_auc = pq_cmp
                    best_val_bce = val_bce
                    best_state = copy.deepcopy(model.state_dict())
                    self.best_epoch_ = epoch
            else:
                if val_bce < best_val_bce:
                    best_val_bce = val_bce
                    best_state = copy.deepcopy(model.state_dict())
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
            P_val = model(zero_val_U, val_q_t, zero_val_V, val_l_t).cpu().numpy()
        self.val_bce_ = _bce(P_val, S_val)
        self.val_top1_ = _top1_acc(P_val, S_val)
        if self.val_objective == "pq_auc":
            self.val_pq_auc_ = _pq_auc(P_val, S_val)

        self.zero_branch_exposure_ = {
            "configured_p_drop_q": self.p_drop_q,
            "configured_p_drop_m": self.p_drop_m,
            "realized_frac_q_dropped": zero_q_count / zero_q_total,
            "realized_frac_m_dropped": zero_m_count / zero_m_total,
        }

        # ── Four-cell diagnostic on S_train (warm / cold-query / cold-model /
        # cold-cold), forcing each of the 4 U/V on-or-zeroed combinations
        # globally over the train pool. Synthesizes the taxonomy from
        # S_train alone, per design doc §1. ─────────────────────────────
        zero_U = torch.zeros_like(U)
        zero_V = torch.zeros_like(V)
        cell_diag = {}
        with torch.no_grad():
            for u_on, m_on, name in [
                (True, True, "warm"),
                (True, False, "cold_query"),
                (False, True, "cold_model"),
                (False, False, "cold_cold"),
            ]:
                U_in = U if u_on else zero_U
                V_in = V if m_on else zero_V
                P = model(U_in, train_q, V_in, train_l).cpu().numpy()
                cell_diag[name] = {
                    "bce": _bce(P, S_train),
                    "top1_acc": _top1_acc(P, S_train),
                    "pq_auc": _pq_auc(P, S_train),
                }
        self.cell_diag_ = cell_diag

        if self.verbose:
            print(
                f"[DropoutNetRouter] best_epoch={self.best_epoch_}  "
                f"val_bce={self.val_bce_:.5f}  val_top1={self.val_top1_:.4f}  "
                f"mf_train_bce={self.mf_train_bce_:.5f}  "
                f"zero_frac_q={self.zero_branch_exposure_['realized_frac_q_dropped']:.3f}  "
                f"zero_frac_m={self.zero_branch_exposure_['realized_frac_m_dropped']:.3f}  "
                f"total={time.time()-t0:.1f}s"
            )
            for name, m in cell_diag.items():
                print(
                    f"  four_cell[{name:11s}]  bce={m['bce']:.4f}  "
                    f"top1={m['top1_acc']:.4f}  pq_auc={m['pq_auc']:.4f}"
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
        U_zero = torch.zeros(q.shape[0], self.mf_dim, device=self._device)
        V_zero = torch.zeros(l.shape[0], self.mf_dim, device=self._device)
        self._model.eval()
        with torch.no_grad():
            P = self._model(U_zero, q, V_zero, l).cpu().numpy()
        return P.astype(np.float32)
