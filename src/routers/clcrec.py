"""
clcrec.py — CLCRec-style joint-contrastive cold-start router.

Adapted from Wei, Wang, Nie, He, Chua, "Contrastive Learning for Cold-Start
Recommendation" (ACM MM 2021); see experiments/router-eval/dropoutnet-clcrec-baselines/README.md.
Unlike DropoutNet, there is no separate pretrain stage: the query encoder
g_q(x_q), the LLM content encoder f_m(y_m), and a train-pool-only
collaborative LLM embedding table Z are all trained jointly, from random
init, with three loss terms per step:

  L_RE  — self-discrimination InfoNCE between Z_m and f_m(y_m), in-batch
          (full train-pool) negatives. Doubles as a per-model Assumption-1
          diagnostic: a model whose L_RE stays high after training has a
          fingerprint that doesn't separate it from other models.
  L_UI  — interaction-level contrastive over observed S_train pairs, with
          EXHAUSTIVE negatives from M_train (not upstream ColdRec's sampled
          num_neg negatives — the model pool here is small, tens to hundreds,
          so exhaustive is affordable and is the design doc's explicit §4
          recommendation).
  L_BCE — sigmoid(s * g_q(x_q) . f_m(y_m) + b) against real S_train. This is
          the head the routing decision rule actually reads from at predict
          time (see _CLCRecModel docstring for why Z is excluded from it).

loss = lambda_re * L_RE + (1 - lambda_re) * L_UI + w_bce * L_BCE

Encoder architecture mirrors src/routers/mlp_dot.py (MLPDotRouter):
n_layers x [Linear -> BatchNorm1d -> ReLU -> Dropout] + Linear -> L2-normalize.
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


# ── Internal helpers (duplicated — see dropoutnet.py / mlp_dot.py for why
# each router file in this codebase is self-contained) ─────────────────────

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


class _CLCRecModel(nn.Module):
    """g_q(x_q), f_m(y_m) content encoders + a train-pool-only Z table.

    Head-path decision (pins down the design doc's looser "z_m <-> jointly
    learned model embedding" phrasing to one unambiguous code path): the BCE
    score that the routing decision rule actually reads from is
    sigmoid(s * g_q(x_q) . f_m(y_m) + b) — the pure CONTENT path, identical
    at train and predict time for both sides. Z is used ONLY inside L_RE, as
    the anchor f_m(y_m) is trained to align with; it never appears in the
    BCE score or in predict(). This is what makes predict() well-defined for
    Q_test/M_test, which have no Z row — matches CLCRec's own inference
    behavior (a cold item's score comes directly from its content embedding).
    """

    def __init__(
        self,
        query_dim: int,
        llm_dim: int,
        n_train_llms: int,
        target_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float,
        normalize_output: bool,
        temperature_init: float,
        learn_temperature: bool,
    ):
        super().__init__()
        self.g_q = _MLPEncoder(
            input_dim=query_dim,
            target_dim=target_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            normalize_output=normalize_output,
        )
        self.f_m = _MLPEncoder(
            input_dim=llm_dim,
            target_dim=target_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            normalize_output=normalize_output,
        )
        self.Z = nn.Embedding(n_train_llms, target_dim)
        nn.init.normal_(self.Z.weight, std=0.01)

        self.log_s = nn.Parameter(
            torch.tensor(math.log(temperature_init), dtype=torch.float32),
            requires_grad=learn_temperature,
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def score(self, z_q: torch.Tensor, z_m: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(torch.exp(self.log_s) * (z_q @ z_m.T) + self.bias)

    def forward(self, q: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        return self.score(self.g_q(q), self.f_m(l))


# ── Router ────────────────────────────────────────────────────────────────────

class CLCRecRouter:
    """CLCRec-style joint-contrastive cold-start router.

    Sklearn-style contract identical to MLPDotRouter/DropoutNetRouter:
        fit(train_q, train_l, S_train, val_q, val_l, S_val) -> self
        predict(test_q, test_l) -> (Nq_test, Nl_test) float32 in [0, 1]

    fit()'s signature is intentionally kept at exactly 6 parameters — no
    train_llm_names kwarg — because src/eval/runner.py::_fit_router
    dispatches on `len(inspect.signature(router.fit).parameters) >= 6` to
    decide whether to pass the val split; a 7th parameter would silently
    break that dispatch. Consequently, self.re_loss_per_model_ (see below)
    is indexed by train_l_emb's row order, not by name — the caller zips it
    against LLM names externally (same pattern src/eval/report.py uses for
    test_task_ids, which is also passed in separately from the router).

    Parameters
    ----------
    lambda_re : float
        Weight on L_RE vs. (1 - lambda_re) on L_UI.
    tau : float
        InfoNCE temperature, shared between L_RE and L_UI (a simplification;
        split into two temperatures later if per-diagnostic tuning calls
        for it).
    w_bce : float
        Weight on the auxiliary BCE head loss, added on top of the
        contrastive terms (contrastive terms as representation-learning
        regularizers, BCE as the calibrated scorer — design doc §6).
    target_dim, hidden_dim, n_layers, dropout, normalize_output,
    temperature_init, learn_temperature, lr, weight_decay, epochs,
    batch_size, seed, device, verbose, val_objective
        Same meaning as MLPDotRouter.
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
        lambda_re: float = 0.5,
        tau: float = 0.1,
        w_bce: float = 1.0,
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
        self.lambda_re = lambda_re
        self.tau = tau
        self.w_bce = w_bce
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
        self.re_loss_per_model_: np.ndarray | None = None
        self._model: _CLCRecModel | None = None
        self._device: torch.device | None = None

    def fit(
        self,
        train_q_emb: np.ndarray,   # (Nq_train, d_q)  float32
        train_l_emb: np.ndarray,   # (Nl_train, d_l)  float32
        S_train: np.ndarray,       # (Nq_train, Nl_train) {0,1}
        val_q_emb: np.ndarray,     # (Nq_val, d_q)
        val_l_emb: np.ndarray,     # (Nl_val, d_l)
        S_val: np.ndarray,         # (Nq_val, Nl_val) {0,1}
    ) -> "CLCRecRouter":
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

        n_q = train_q.shape[0]
        n_l = train_l.shape[0]

        model = _CLCRecModel(
            query_dim=train_q.shape[1],
            llm_dim=train_l.shape[1],
            n_train_llms=n_l,
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

        diag_labels = torch.arange(n_l, device=device)
        best_val_bce = float("inf")
        best_val_pq_auc = -1.0
        best_state: dict | None = None

        if self.verbose:
            print(
                f"[CLCRecRouter] fit  device={device}  "
                f"train={n_q}q x {n_l}l  val={val_q_t.shape[0]}q x {val_l_t.shape[0]}l  "
                f"target_dim={self.target_dim}  epochs={self.epochs}  "
                f"lambda_re={self.lambda_re}  tau={self.tau}  w_bce={self.w_bce}"
            )

        t0 = time.time()
        for epoch in range(1, self.epochs + 1):
            model.train()
            perm = torch.randperm(n_q, device=device)

            epoch_loss = 0.0
            epoch_re = 0.0
            epoch_ui = 0.0
            epoch_bce = 0.0
            n_batches = 0
            for start in range(0, n_q, self.batch_size):
                idx = perm[start : start + self.batch_size]
                q_batch = train_q[idx]        # (B, d_q)
                s_batch = S_train_t[idx]      # (B, Nl_train)

                optimizer.zero_grad()

                z_q = model.g_q(q_batch)              # (B, D)
                f_m_all = model.f_m(train_l)          # (Nl_train, D)
                Z_all = model.Z.weight                # (Nl_train, D)

                # ── L_RE: self-discrimination InfoNCE, Z <-> f_m(y_m) ────
                logits_re = (Z_all @ f_m_all.T) / self.tau     # (Nl_train, Nl_train)
                L_RE = F.cross_entropy(logits_re, diag_labels)

                # ── L_UI: interaction contrastive, exhaustive negatives ──
                s_val = torch.exp(model.log_s)
                logits_ui = (s_val * (z_q @ f_m_all.T) + model.bias) / self.tau  # (B, Nl_train)
                logsumexp_row = torch.logsumexp(logits_ui, dim=1, keepdim=True)  # (B, 1)
                per_pair_loss = logsumexp_row - logits_ui                        # (B, Nl_train)
                pos_count = s_batch.sum().clamp(min=1.0)
                L_UI = (per_pair_loss * s_batch).sum() / pos_count

                # ── BCE head (the actual scorer) ─────────────────────────
                scores = model.score(z_q, f_m_all)
                L_BCE = F.binary_cross_entropy(scores, s_batch)

                loss = self.lambda_re * L_RE + (1 - self.lambda_re) * L_UI + self.w_bce * L_BCE
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_re += L_RE.item()
                epoch_ui += L_UI.item()
                epoch_bce += L_BCE.item()
                n_batches += 1

            # ── Val evaluation (content path only, no Z needed) ─────────
            model.eval()
            with torch.no_grad():
                P_val = model(val_q_t, val_l_t).cpu().numpy()

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
                    f"loss={epoch_loss/n_batches:.4f}  "
                    f"L_RE={epoch_re/n_batches:.4f}  L_UI={epoch_ui/n_batches:.4f}  "
                    f"L_BCE={epoch_bce/n_batches:.4f}  "
                    f"val_bce={val_bce:.4f}  val_top1={val_top1:.4f}  "
                    f"[{time.time()-t0:.0f}s]"
                )

        # ── Restore best checkpoint ───────────────────────────────────────
        assert best_state is not None
        model.load_state_dict(best_state)
        model.eval()
        self._model = model

        with torch.no_grad():
            P_val = model(val_q_t, val_l_t).cpu().numpy()
        self.val_bce_ = _bce(P_val, S_val)
        self.val_top1_ = _top1_acc(P_val, S_val)
        if self.val_objective == "pq_auc":
            self.val_pq_auc_ = _pq_auc(P_val, S_val)

        # ── Per-model L_RE-at-convergence: Assumption-1 diagnostic ───────
        with torch.no_grad():
            f_m_all = model.f_m(train_l)
            Z_all = model.Z.weight
            logits_re = (Z_all @ f_m_all.T) / self.tau
            per_model_ce = F.cross_entropy(logits_re, diag_labels, reduction="none")
        self.re_loss_per_model_ = per_model_ce.cpu().numpy().astype(np.float32)

        if self.verbose:
            print(
                f"[CLCRecRouter] best_epoch={self.best_epoch_}  "
                f"val_bce={self.val_bce_:.5f}  val_top1={self.val_top1_:.4f}  "
                f"re_loss_per_model  mean={self.re_loss_per_model_.mean():.4f}  "
                f"std={self.re_loss_per_model_.std():.4f}  "
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
            P = self._model(q, l).cpu().numpy()
        return P.astype(np.float32)
