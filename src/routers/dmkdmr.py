"""
dmkdmr.py — Discriminative Memory-KDM Router (DMKDMR).

Sklearn-style port of router-eval/src/methods/MKDMR/dmkdmr.py.

For each (query, candidate-LLM) pair, estimates

    P(correct=1 | c, q) = score_pos(c|q) / (score_pos + score_neg)(c|q)

via a double RBF kernel over (query, model) pairs from the training memory.
The denominator factorises:

    denom(c | q) = A_sum(q) · K_y_colsum(c)

so routing N_test queries over K_cand candidates requires no per-candidate
inner loop — just one (N_test, k) × (k, K_cand) expression per batch.

V_pos — the only (N_train, K_cand) matrix — is built once per σ_y via a
sparse mat-mul against the binary correctness matrix S_train.  Bandwidths
(σ_x, σ_y) are tuned by nested Brent search against a val BCE objective,
anchored to median pairwise L2 distances.

Pool mapping (cold-model)
-------------------------
Memory pool : train LLMs  (columns of S_train)
Candidates  : whatever predict() receives — open-pool by construction.
"""

import math
from typing import Literal

import numpy as np
import scipy.optimize
import scipy.sparse
import torch
from sklearn.neighbors import NearestNeighbors


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cosine_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return X / norms


def _sigma_median(points: np.ndarray, n_sample: int = 1000, seed: int = 0) -> float:
    """Median pairwise L2 distance on a random subsample — RBF bandwidth heuristic."""
    rng = np.random.default_rng(seed)
    sub = points[rng.choice(len(points), size=min(n_sample, len(points)), replace=False)]
    nn = NearestNeighbors(n_neighbors=len(sub) - 1, algorithm="brute")
    nn.fit(sub)
    dists, _ = nn.kneighbors(sub)
    return float(np.median(dists))


def _build_knn_data(
    query_embed: np.ndarray,
    train_embed: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (knn_idx, knn_sqd) for cosine-normalised embeddings.

    Euclidean distance in cosine-normalised space equals sqrt(2 - 2·cos_sim),
    so exact cosine k-NN is recovered without a cosine-distance call.
    """
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="brute")
    nn.fit(train_embed)
    dists, idx = nn.kneighbors(query_embed)
    return idx.astype(np.int32), (dists ** 2).astype(np.float32)


def _build_V_pos_and_colsum(
    train_score_csr: scipy.sparse.csr_matrix,  # (N_train, M_full) binary
    model_embed_full: torch.Tensor,             # (M_full, D)  on device
    candidate_embeds: torch.Tensor,             # (K_cand, D)  on device
    sigma_y: float,
    kernel_batch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the positive-class density summary and total kernel mass.

    Returns
    -------
    V_pos      : (N_train, K_cand) float32  CPU
        V_pos[i, c] = Σ_{j ∈ pos(Q_i)} exp(−‖M[j] − cand[c]‖² / 2σ_y²)
    K_y_colsum : (K_cand,) float32  CPU
        K_y_colsum[c] = Σ_j exp(−‖M[j] − cand[c]‖² / 2σ_y²)

    Batches over K_cand (chunks of `kernel_batch`) to avoid the
    (M_full, K_cand, D) intermediate tensor.  Peak per-batch memory:
    M_full × kernel_batch × 4 bytes (e.g. 3k × 256 × 4 B = 3 MB).
    Uses ‖a‖² + ‖b‖² − 2·a·bᵀ for efficient pairwise squared distances.
    """
    N_train = train_score_csr.shape[0]
    K_cand  = candidate_embeds.shape[0]

    V_pos_np   = np.empty((N_train, K_cand), dtype=np.float32)
    K_y_colsum = np.empty(K_cand, dtype=np.float32)

    M_norm2 = (model_embed_full ** 2).sum(-1)  # (M,)

    for c0 in range(0, K_cand, kernel_batch):
        cb    = candidate_embeds[c0 : c0 + kernel_batch]          # (Kb, D)
        cb2   = (cb ** 2).sum(-1)                                  # (Kb,)
        sqd   = M_norm2[:, None] + cb2[None, :] - 2.0 * (model_embed_full @ cb.T)  # (M, Kb)
        K_y_b = torch.exp(-sqd / (2.0 * sigma_y ** 2))            # (M, Kb)
        K_y_b_np = K_y_b.cpu().numpy()
        V_pos_np[:, c0 : c0 + kernel_batch] = train_score_csr.dot(K_y_b_np)
        K_y_colsum[c0 : c0 + kernel_batch]  = K_y_b_np.sum(axis=0)

    device = model_embed_full.device
    return (
        torch.from_numpy(V_pos_np).to(device),
        torch.from_numpy(K_y_colsum).to(device),
    )


def _route_posterior(
    knn_idx_t: torch.Tensor,    # (N, k) long
    knn_sqd_t: torch.Tensor,    # (N, k) float
    V_pos: torch.Tensor,        # (N_train, K_cand) float
    K_y_colsum: torch.Tensor,   # (K_cand,) float
    sigma_x: float,
    batch_size: int,
    eps: float = 1e-12,
) -> np.ndarray:                # (N, K_cand) float32 in [0, 1]
    """Compute P(correct=1 | c, q) for every query.

    denom(c|q) = A_sum(q) · K_y_colsum(c) factorises, so the per-query
    cost is dominated by one (B, k, K_cand) gather + einsum per batch.
    For K_cand > ~10k, a scatter→matmul reformulation (dense W of shape
    (B, N_train) then W @ V_pos) would halve peak memory.
    """
    N      = knn_idx_t.shape[0]
    K_cand = V_pos.shape[1]
    out    = torch.empty(N, K_cand, device=V_pos.device)

    for start in range(0, N, batch_size):
        end   = min(start + batch_size, N)
        idx_b = knn_idx_t[start:end]                              # (B, k)
        sqd_b = knn_sqd_t[start:end]                              # (B, k)

        A         = torch.exp(-sqd_b / (2.0 * sigma_x ** 2))     # (B, k)
        Vg        = V_pos[idx_b]                                  # (B, k, K)
        score_pos = torch.einsum("bk,bkc->bc", A, Vg)             # (B, K)

        A_sum = A.sum(dim=1, keepdim=True)                        # (B, 1)
        denom = A_sum * K_y_colsum.unsqueeze(0)                   # (B, K)

        out[start:end] = score_pos / (denom + eps)

    P = out.cpu().numpy()
    assert np.all(P >= -1e-5) and np.all(P <= 1.0 + 1e-5), (
        f"Posterior out of [0,1]: min={P.min():.4f} max={P.max():.4f}"
    )
    return np.clip(P, 0.0, 1.0).astype(np.float32)


def _val_bce(P_correct: np.ndarray, val_score: np.ndarray, eps: float = 1e-7) -> float:
    """Mean binary cross-entropy over all (q, c) val pairs."""
    p = np.clip(P_correct, eps, 1.0 - eps)
    y = val_score.astype(np.float32)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def _val_pq_auc(P_correct: np.ndarray, val_score: np.ndarray) -> float:
    """Mean per-query ROC AUC of P_correct vs val_score.

    Skips queries with degenerate labels (all-correct or all-wrong). Returns
    NaN if every row is degenerate.
    """
    from sklearn.metrics import roc_auc_score
    aucs = []
    for i in range(val_score.shape[0]):
        y = val_score[i]
        if y.min() == y.max():
            continue
        aucs.append(roc_auc_score(y, P_correct[i]))
    return float(np.mean(aucs)) if aucs else float("nan")


def _val_mse(P_correct: np.ndarray, val_score: np.ndarray) -> float:
    """Mean squared error between predicted posteriors and val scores (Brier score)."""
    return float(((P_correct - val_score.astype(np.float32)) ** 2).mean())


# ── Router class ──────────────────────────────────────────────────────────────

class DMKDMRRouter:
    """Discriminative Memory-KDM router.

    Predicts P(correct=1 | candidate c, query q) via a double-RBF kernel
    density over (query, model) pairs from the training memory, applying
    Bayes' rule per candidate.  Calibrated probabilities in [0, 1].

    Bandwidths σ_x (query space) and σ_y (model space) are tuned by nested
    Brent search against a val objective (≈10 outer × 10 inner evals).

    Parameters
    ----------
    k : int
        k-NN neighbours retrieved per test query from the training set.
    sigma_x_mult_range, sigma_y_mult_range : (float, float)
        Brent bounds as multipliers of the median heuristic.
    tol_x, tol_y : float
        Brent xatol in log-multiplier units.
    objective : {"bce", "top1_acc", "pq_auc", "mse"}
        Brent minimization target. "bce" minimizes mean BCE (default);
        "top1_acc" maximizes val top-1 accuracy (Brent receives -accuracy);
        "pq_auc" maximizes mean per-query ROC AUC (Brent receives -AUC);
        "mse" minimizes mean squared error (Brier score) — faster than pq_auc.
        val_bce_, val_top1_, val_pq_auc_, and val_mse_ are always populated after fit().
    batch_size : int
        Query batch size in _route_posterior.
    kernel_batch : int
        Candidate batch size in _build_V_pos_and_colsum.
        Peak memory per batch ≈ M_full × kernel_batch × 4 B.
    seed : int
        Seed for median-heuristic subsampling.
    device : str | None
        Torch device. None → "cuda" if available, else "cpu".
    verbose : bool
        Print Brent progress to stdout.
    """

    def __init__(
        self,
        k: int = 128,
        sigma_x_mult_range: tuple[float, float] = (0.0625, 4.0),
        sigma_y_mult_range: tuple[float, float] = (0.0625, 4.0),
        tol_x: float = 0.05,
        tol_y: float = 0.05,
        objective: Literal["bce", "top1_acc", "pq_auc", "mse"] = "bce",
        batch_size: int = 256,
        kernel_batch: int = 256,
        seed: int = 0,
        device: str | None = None,
        verbose: bool = True,
    ) -> None:
        assert objective in ("bce", "top1_acc", "pq_auc", "mse"), \
            f"Unknown objective: {objective!r}"
        self.k = k
        self.sigma_x_mult_range = sigma_x_mult_range
        self.sigma_y_mult_range = sigma_y_mult_range
        self.tol_x = tol_x
        self.tol_y = tol_y
        self.objective = objective
        self.batch_size = batch_size
        self.kernel_batch = kernel_batch
        self.seed = seed
        self.device = device
        self.verbose = verbose

        # Set by fit()
        self.sigma_x_: float | None = None
        self.sigma_y_: float | None = None
        self.val_bce_: float | None = None
        self.val_top1_: float | None = None
        self.val_pq_auc_: float | None = None
        self.val_mse_: float | None = None
        self.val_obj_: float | None = None
        self._train_q_emb: np.ndarray | None = None
        self._train_l_emb: np.ndarray | None = None
        self._train_score_csr: scipy.sparse.csr_matrix | None = None
        self._device: torch.device | None = None

    def fit(
        self,
        train_q_emb: np.ndarray,   # (Nq_train, d_q)  float32
        train_l_emb: np.ndarray,   # (Nl_train, d_l)  float32  — memory pool
        S_train: np.ndarray,       # (Nq_train, Nl_train) {0,1}
        val_q_emb: np.ndarray,     # (Nq_val,   d_q)
        val_l_emb: np.ndarray,     # (Nl_val,   d_l)  — val candidates
        S_val: np.ndarray,         # (Nq_val,   Nl_val) {0,1}
    ) -> "DMKDMRRouter":
        """Cosine-normalise, compute median bandwidths, run nested Brent search."""
        np.random.seed(self.seed)
        device = torch.device(
            self.device if self.device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._device = device

        train_q = _cosine_normalize(train_q_emb.astype(np.float32))
        train_l = _cosine_normalize(train_l_emb.astype(np.float32))
        val_q   = _cosine_normalize(val_q_emb.astype(np.float32))
        val_l   = _cosine_normalize(val_l_emb.astype(np.float32))

        self._train_q_emb     = train_q
        self._train_l_emb     = train_l
        self._train_score_csr = scipy.sparse.csr_matrix(S_train)

        sigma_x_med = _sigma_median(train_q, seed=self.seed)
        sigma_y_med = _sigma_median(train_l, seed=self.seed)
        self._sigma_x_med = sigma_x_med
        self._sigma_y_med = sigma_y_med

        if self.verbose:
            print(f"[DMKDMR fit]  N_train_q={len(train_q)}  Nl_train={len(train_l)}  "
                  f"N_val_q={len(val_q)}  Nl_val={len(val_l)}  device={device}")
            print(f"[DMKDMR fit]  sigma_x_med={sigma_x_med:.4f}  sigma_y_med={sigma_y_med:.4f}")

        log_mx_lo = math.log(self.sigma_x_mult_range[0])
        log_mx_hi = math.log(self.sigma_x_mult_range[1])
        log_my_lo = math.log(self.sigma_y_mult_range[0])
        log_my_hi = math.log(self.sigma_y_mult_range[1])

        if self.verbose:
            print(f"[DMKDMR fit]  building val k-NN (k={self.k}) ...")
        val_knn_idx, val_knn_sqd = _build_knn_data(val_q, train_q, self.k)
        val_knn_idx_t = torch.from_numpy(val_knn_idx).long().to(device)
        val_knn_sqd_t = torch.from_numpy(val_knn_sqd).to(device)
        val_score_f32 = S_val.astype(np.float32)

        train_l_t = torch.from_numpy(train_l).to(device)
        val_l_t   = torch.from_numpy(val_l).to(device)

        outer_call = [0]

        def inner_obj(log_mx, V_pos, K_y_colsum):
            sigma_x = sigma_x_med * math.exp(log_mx)
            P = _route_posterior(
                val_knn_idx_t, val_knn_sqd_t,
                V_pos, K_y_colsum, sigma_x, self.batch_size,
            )
            if self.objective == "bce":
                obj = _val_bce(P, val_score_f32)
            elif self.objective == "top1_acc":
                picks = P.argmax(axis=1)
                acc = float(val_score_f32[np.arange(P.shape[0]), picks].mean())
                obj = -acc
            elif self.objective == "pq_auc":
                obj = -_val_pq_auc(P, val_score_f32)
            else:  # "mse"
                obj = _val_mse(P, val_score_f32)
            if self.verbose:
                print(f"    [inner]  sigma_x={sigma_x:.5f} (x{math.exp(log_mx):.4f})"
                      f"  val_obj={obj:.5f}")
            return obj

        def outer_obj(log_my):
            outer_call[0] += 1
            sigma_y = sigma_y_med * math.exp(log_my)
            if self.verbose:
                print(f"  [outer #{outer_call[0]}]  sigma_y={sigma_y:.5f} "
                      f"(x{math.exp(log_my):.4f})  building V_pos ...")
            V_pos, K_y_colsum = _build_V_pos_and_colsum(
                self._train_score_csr, train_l_t, val_l_t,
                sigma_y, self.kernel_batch,
            )
            res = scipy.optimize.minimize_scalar(
                inner_obj,
                args=(V_pos, K_y_colsum),
                bounds=(log_mx_lo, log_mx_hi),
                method="bounded",
                options={"xatol": self.tol_x},
            )
            if self.verbose:
                best_sx = sigma_x_med * math.exp(res.x)
                print(f"  [outer #{outer_call[0]}]  best sigma_x={best_sx:.5f}"
                      f"  val_obj={res.fun:.5f}")
            return res.fun

        if self.verbose:
            print("[DMKDMR fit]  outer Brent search over sigma_y ...")
        outer_res = scipy.optimize.minimize_scalar(
            outer_obj,
            bounds=(log_my_lo, log_my_hi),
            method="bounded",
            options={"xatol": self.tol_y},
        )

        best_sigma_y = sigma_y_med * math.exp(outer_res.x)

        # Rebuild V_pos at σ_y* and re-solve inner to recover the correct σ_x*.
        # The inner argmin returned mid-Brent may correspond to a different σ_y.
        if self.verbose:
            print(f"[DMKDMR fit]  final rebuild at sigma_y={best_sigma_y:.5f} ...")
        V_pos_final, K_y_colsum_final = _build_V_pos_and_colsum(
            self._train_score_csr, train_l_t, val_l_t,
            best_sigma_y, self.kernel_batch,
        )
        final_inner = scipy.optimize.minimize_scalar(
            inner_obj,
            args=(V_pos_final, K_y_colsum_final),
            bounds=(log_mx_lo, log_mx_hi),
            method="bounded",
            options={"xatol": self.tol_x},
        )
        best_sigma_x = sigma_x_med * math.exp(final_inner.x)

        self.sigma_x_ = best_sigma_x
        self.sigma_y_ = best_sigma_y
        self.val_obj_  = final_inner.fun

        # Always compute both metrics at the chosen sigmas for downstream consumers.
        P_final = _route_posterior(
            val_knn_idx_t, val_knn_sqd_t,
            V_pos_final, K_y_colsum_final,
            best_sigma_x, self.batch_size,
        )
        self.val_bce_    = _val_bce(P_final, val_score_f32)
        _picks           = P_final.argmax(axis=1)
        self.val_top1_   = float(val_score_f32[np.arange(P_final.shape[0]), _picks].mean())
        self.val_pq_auc_ = _val_pq_auc(P_final, val_score_f32)
        self.val_mse_    = _val_mse(P_final, val_score_f32)

        if self.verbose:
            print(f"[DMKDMR fit]  sigma_x={best_sigma_x:.5f}  "
                  f"sigma_y={best_sigma_y:.5f}  val_bce={self.val_bce_:.5f}  "
                  f"val_top1={self.val_top1_:.5f}  "
                  f"val_pq_auc={self.val_pq_auc_:.5f}  "
                  f"val_mse={self.val_mse_:.5f}  outer_evals={outer_call[0]}")
        return self

    def predict(
        self,
        test_q_emb: np.ndarray,   # (Nq_test, d_q)
        test_l_emb: np.ndarray,   # (Nl_test, d_l)  — candidates
    ) -> np.ndarray:              # (Nq_test, Nl_test) float32 in [0, 1]
        """Route test queries among test LLM candidates.

        Rebuilds V_pos / K_y_colsum for test_l_emb at the tuned σ_y*.
        Returns calibrated P(correct=1 | candidate, query) in [0, 1].
        """
        if self.sigma_x_ is None:
            raise RuntimeError("Call fit() before predict().")

        device = self._device
        test_q = _cosine_normalize(test_q_emb.astype(np.float32))
        test_l = _cosine_normalize(test_l_emb.astype(np.float32))

        if self.verbose:
            print(f"[DMKDMR predict]  building test k-NN (k={self.k}, "
                  f"Nq_test={len(test_q)}, Nl_test={len(test_l)}) ...")
        test_knn_idx, test_knn_sqd = _build_knn_data(test_q, self._train_q_emb, self.k)
        test_knn_idx_t = torch.from_numpy(test_knn_idx).long().to(device)
        test_knn_sqd_t = torch.from_numpy(test_knn_sqd).to(device)

        train_l_t = torch.from_numpy(self._train_l_emb).to(device)
        test_l_t  = torch.from_numpy(test_l).to(device)

        if self.verbose:
            print(f"[DMKDMR predict]  building V_pos at sigma_y={self.sigma_y_:.5f} ...")
        V_pos, K_y_colsum = _build_V_pos_and_colsum(
            self._train_score_csr, train_l_t, test_l_t,
            self.sigma_y_, self.kernel_batch,
        )

        return _route_posterior(
            test_knn_idx_t, test_knn_sqd_t,
            V_pos, K_y_colsum,
            self.sigma_x_, self.batch_size,
        )
