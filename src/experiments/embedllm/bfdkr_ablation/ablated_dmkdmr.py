"""
ablated_dmkdmr.py — AblatedDMKDMRRouter for exp_22 component knockouts.

Three ablated variants of DMKDMRRouter, each realised as an extreme-bandwidth
limit of the *unchanged* KDM estimator:

  qk_off  query kernel killed  (σ_x → ∞  ⟹ uniform A over k neighbours)
  lk_off  LLM kernel killed    (σ_y → 0⁺ ⟹ hard 1-NN borrow via stabilised K̃_y)
  bw_off  both bandwidths fixed at median heuristic (no Brent search)

Stabilised kernel
-----------------
K̃_y(j, c) = exp( -(d²_jc - min_{j'} d²_{j'c}) / (2 σ_y²) )

The per-candidate-column shift is a multiplicative rescaling by
β_c = exp(min_{j'} d²_{j'c} / (2 σ_y²)), which cancels exactly in the Bayes
posterior ratio — V_pos[:, c] and K_y_colsum[c] both carry β_c:

    P̂(1 | c, q) = Σ_{i∈N(q)} A_i · V_pos[i,c]   /   A_sum · K_y_colsum[c]

So for any finite σ_y the posterior is numerically identical to the base kernel.
At σ_y → 0 the nearest memory LLM holds exp(0) = 1, all others underflow to 0 →
exact hard 1-NN borrow, no 0/0.

src/routers/dmkdmr.py is NOT modified.  Base DMKDMRRouter stays byte-identical so
the exp_20 reused anchor remains valid.
"""

import math
from typing import Literal

import numpy as np
import scipy.optimize
import scipy.sparse
import torch

from src.routers.dmkdmr import (
    DMKDMRRouter,
    _build_knn_data,
    _cosine_normalize,
    _route_posterior,
    _sigma_median,
    _val_bce,
    _val_mse,
    _val_pq_auc,
)

# σ_y multiplier for the lk_off hard-1NN limit.
# exp(-d² / (2 σ²)) with σ = σ_med · 1e-6 and d² ≤ 4 gives
# exp(-4 / (2 · (σ_med · 1e-6)²)) ≈ 0 for any σ_med ≥ 0.01.
_LK_OFF_SIGMA_Y_MULT: float = 1e-6


# ── Stabilised kernel build ────────────────────────────────────────────────────

def _build_V_pos_and_colsum_stabilised(
    train_score_csr: scipy.sparse.csr_matrix,
    model_embed_full: torch.Tensor,       # (M, D)  cosine-normalised, on device
    candidate_embeds: torch.Tensor,       # (K_cand, D) cosine-normalised, on device
    sigma_y: float,
    kernel_batch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stabilised V_pos and K_y_colsum (see module docstring).

    Identical to src.routers.dmkdmr._build_V_pos_and_colsum except for
    the per-candidate min-subtraction before the exponent.

    Returns
    -------
    V_pos      : (N_train, K_cand) float32  on device
    K_y_colsum : (K_cand,) float32  on device
    """
    N_train = train_score_csr.shape[0]
    K_cand  = candidate_embeds.shape[0]

    V_pos_np   = np.empty((N_train, K_cand), dtype=np.float32)
    K_y_colsum = np.empty(K_cand, dtype=np.float32)

    M_norm2 = (model_embed_full ** 2).sum(-1)   # (M,)

    for c0 in range(0, K_cand, kernel_batch):
        cb   = candidate_embeds[c0 : c0 + kernel_batch]          # (Kb, D)
        cb2  = (cb ** 2).sum(-1)                                  # (Kb,)
        sqd  = M_norm2[:, None] + cb2[None, :] - 2.0 * (model_embed_full @ cb.T)
        sqd  = sqd.clamp(min=0.0)           # guard tiny fp negatives

        # Per-candidate min-subtract: shape (1, Kb) broadcasts over (M, Kb).
        # β_c cancels in numerator / denominator → posterior unchanged.
        sqd_s = sqd - sqd.min(dim=0, keepdim=True).values          # (M, Kb)

        K_y_b    = torch.exp(-sqd_s / (2.0 * sigma_y ** 2))       # (M, Kb)
        K_y_b_np = K_y_b.cpu().numpy()
        V_pos_np[:, c0 : c0 + kernel_batch] = train_score_csr.dot(K_y_b_np)
        K_y_colsum[c0 : c0 + kernel_batch]  = K_y_b_np.sum(axis=0)

    device = model_embed_full.device
    return (
        torch.from_numpy(V_pos_np).to(device),
        torch.from_numpy(K_y_colsum).to(device),
    )


# ── Router subclass ────────────────────────────────────────────────────────────

class AblatedDMKDMRRouter(DMKDMRRouter):
    """DMKDMRRouter with one component knocked out.

    Parameters
    ----------
    knockout : {"qk_off", "lk_off", "bw_off"}
        qk_off  query kernel off: σ_x → ∞ (uniform neighbour average); σ_y Brent-tuned.
        lk_off  LLM kernel off:   σ_y → 0⁺ (hard 1-NN via stabilised kernel); σ_x Brent-tuned.
        bw_off  both bandwidths pinned to median heuristic; no Brent.
    All other kwargs forwarded to DMKDMRRouter.

    Notes
    -----
    After fit(), val_pq_auc_, val_mse_, val_top1_, sigma_x_, sigma_y_ are always
    set (mandatory for the driver k-selection and JSON output).

    sigma_y_ is stored as the *semantic* value (0.0 for lk_off; tuned σ_y for
    qk_off; median for bw_off).  The *kernel* value used in predict() is held in
    self._sigma_y_kernel.
    """

    def __init__(
        self,
        knockout: Literal["qk_off", "lk_off", "bw_off"],
        **kwargs,
    ) -> None:
        assert knockout in ("qk_off", "lk_off", "bw_off"), \
            f"Unknown knockout: {knockout!r}"
        super().__init__(**kwargs)
        self.knockout = knockout
        self._sigma_y_kernel: float | None = None

    # ------------------------------------------------------------------
    # fit()
    # ------------------------------------------------------------------

    def fit(
        self,
        train_q_emb: np.ndarray,   # (Nq_train, d_q)
        train_l_emb: np.ndarray,   # (Nl_train, d_l)
        S_train: np.ndarray,       # (Nq_train, Nl_train) {0,1}
        val_q_emb: np.ndarray,     # (Nq_val, d_q)
        val_l_emb: np.ndarray,     # (Nl_val, d_l)
        S_val: np.ndarray,         # (Nq_val, Nl_val) {0,1}
    ) -> "AblatedDMKDMRRouter":

        # ── Setup (verbatim from DMKDMRRouter.fit lines 277-316) ──────────────
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
            print(f"[{self.knockout} fit]  N_train_q={len(train_q)}  "
                  f"Nl_train={len(train_l)}  N_val_q={len(val_q)}  "
                  f"Nl_val={len(val_l)}  device={device}")
            print(f"[{self.knockout} fit]  sigma_x_med={sigma_x_med:.4f}  "
                  f"sigma_y_med={sigma_y_med:.4f}")

        if self.verbose:
            print(f"[{self.knockout} fit]  building val k-NN (k={self.k}) ...")
        val_knn_idx, val_knn_sqd = _build_knn_data(val_q, train_q, self.k)
        val_knn_idx_t = torch.from_numpy(val_knn_idx).long().to(device)
        val_knn_sqd_t = torch.from_numpy(val_knn_sqd).to(device)
        val_score_f32 = S_val.astype(np.float32)

        train_l_t = torch.from_numpy(train_l).to(device)
        val_l_t   = torch.from_numpy(val_l).to(device)

        log_mx_lo = math.log(self.sigma_x_mult_range[0])
        log_mx_hi = math.log(self.sigma_x_mult_range[1])
        log_my_lo = math.log(self.sigma_y_mult_range[0])
        log_my_hi = math.log(self.sigma_y_mult_range[1])

        # ── Shared helpers ─────────────────────────────────────────────────────

        def _build_V(sy: float):
            return _build_V_pos_and_colsum_stabilised(
                self._train_score_csr, train_l_t, val_l_t, sy, self.kernel_batch)

        def _eval_obj(P: np.ndarray) -> float:
            if self.objective == "bce":
                return _val_bce(P, val_score_f32)
            elif self.objective == "top1_acc":
                picks = P.argmax(axis=1)
                return -float(val_score_f32[np.arange(P.shape[0]), picks].mean())
            elif self.objective == "pq_auc":
                return -_val_pq_auc(P, val_score_f32)
            else:   # "mse"
                return _val_mse(P, val_score_f32)

        # ── Mode-specific search ───────────────────────────────────────────────

        if self.knockout == "qk_off":
            # σ_x pinned to ∞; outer Brent over σ_y.
            outer_n = [0]

            def _outer_qk(log_my: float) -> float:
                outer_n[0] += 1
                sy = sigma_y_med * math.exp(log_my)
                if self.verbose:
                    print(f"  [qk_off outer #{outer_n[0]}]  sigma_y={sy:.5f}  "
                          "building V_pos ...")
                V, Kcs = _build_V(sy)
                P = _route_posterior(val_knn_idx_t, val_knn_sqd_t,
                                     V, Kcs, float("inf"), self.batch_size)
                obj = _eval_obj(P)
                if self.verbose:
                    print(f"  [qk_off outer #{outer_n[0]}]  val_obj={obj:.5f}")
                return obj

            outer_res = scipy.optimize.minimize_scalar(
                _outer_qk,
                bounds=(log_my_lo, log_my_hi),
                method="bounded",
                options={"xatol": self.tol_y},
            )
            best_sy = sigma_y_med * math.exp(outer_res.x)
            if self.verbose:
                print(f"[qk_off fit]  final rebuild at sigma_y={best_sy:.5f} ...")
            V_f, Kcs_f = _build_V(best_sy)
            P_final = _route_posterior(val_knn_idx_t, val_knn_sqd_t,
                                       V_f, Kcs_f, float("inf"), self.batch_size)
            self.sigma_x_       = float("inf")
            self.sigma_y_       = best_sy
            self._sigma_y_kernel = best_sy

        elif self.knockout == "lk_off":
            # σ_y pinned to 0⁺ limit; single kernel build; inner Brent over σ_x.
            sy_small = sigma_y_med * _LK_OFF_SIGMA_Y_MULT
            if self.verbose:
                print(f"[lk_off fit]  building V_pos at sigma_y_small={sy_small:.2e} ...")
            V_lk, Kcs_lk = _build_V(sy_small)
            inner_n = [0]

            def _inner_lk(log_mx: float) -> float:
                inner_n[0] += 1
                sx = sigma_x_med * math.exp(log_mx)
                P = _route_posterior(val_knn_idx_t, val_knn_sqd_t,
                                     V_lk, Kcs_lk, sx, self.batch_size)
                obj = _eval_obj(P)
                if self.verbose:
                    print(f"  [lk_off inner #{inner_n[0]}]  sigma_x={sx:.5f}  "
                          f"val_obj={obj:.5f}")
                return obj

            inner_res = scipy.optimize.minimize_scalar(
                _inner_lk,
                bounds=(log_mx_lo, log_mx_hi),
                method="bounded",
                options={"xatol": self.tol_x},
            )
            best_sx = sigma_x_med * math.exp(inner_res.x)
            P_final = _route_posterior(val_knn_idx_t, val_knn_sqd_t,
                                       V_lk, Kcs_lk, best_sx, self.batch_size)
            self.sigma_x_        = best_sx
            self.sigma_y_        = 0.0          # semantic: hard-1NN limit
            self._sigma_y_kernel  = sy_small

        else:   # bw_off
            # Both pinned at median; no Brent.
            if self.verbose:
                print(f"[bw_off fit]  building V_pos at sigma_y_med={sigma_y_med:.5f} ...")
            V_bw, Kcs_bw = _build_V(sigma_y_med)
            P_final = _route_posterior(val_knn_idx_t, val_knn_sqd_t,
                                       V_bw, Kcs_bw, sigma_x_med, self.batch_size)
            self.sigma_x_        = sigma_x_med
            self.sigma_y_        = sigma_y_med
            self._sigma_y_kernel  = sigma_y_med

        # ── Record all val metrics (mandatory invariants) ──────────────────────
        self.val_obj_    = float("nan")
        self.val_bce_    = _val_bce(P_final, val_score_f32)
        _picks           = P_final.argmax(axis=1)
        self.val_top1_   = float(val_score_f32[np.arange(P_final.shape[0]), _picks].mean())
        self.val_pq_auc_ = _val_pq_auc(P_final, val_score_f32)
        self.val_mse_    = _val_mse(P_final, val_score_f32)

        if self.verbose:
            print(f"[{self.knockout} fit]  sigma_x={self.sigma_x_}  "
                  f"sigma_y={self.sigma_y_}  "
                  f"val_bce={self.val_bce_:.5f}  val_top1={self.val_top1_:.5f}  "
                  f"val_pq_auc={self.val_pq_auc_:.5f}  val_mse={self.val_mse_:.5f}")
        return self

    # ------------------------------------------------------------------
    # predict() — use stabilised kernel so lk_off works at σ_y → 0
    # ------------------------------------------------------------------

    def predict(
        self,
        test_q_emb: np.ndarray,
        test_l_emb: np.ndarray,
    ) -> np.ndarray:
        """Route test queries using the stabilised K̃_y kernel.

        For qk_off and bw_off the stabilisation is numerically identical to
        the base kernel (β_c cancels in the ratio).  For lk_off it yields the
        hard-1NN posterior without 0/0 NaN.
        """
        if self.sigma_x_ is None:
            raise RuntimeError("Call fit() before predict().")

        device = self._device
        test_q = _cosine_normalize(test_q_emb.astype(np.float32))
        test_l = _cosine_normalize(test_l_emb.astype(np.float32))

        if self.verbose:
            print(f"[{self.knockout} predict]  building test k-NN "
                  f"(k={self.k}, Nq_test={len(test_q)}, Nl_test={len(test_l)}) ...")
        test_knn_idx, test_knn_sqd = _build_knn_data(test_q, self._train_q_emb, self.k)
        test_knn_idx_t = torch.from_numpy(test_knn_idx).long().to(device)
        test_knn_sqd_t = torch.from_numpy(test_knn_sqd).to(device)

        train_l_t = torch.from_numpy(self._train_l_emb).to(device)
        test_l_t  = torch.from_numpy(test_l).to(device)

        if self.verbose:
            print(f"[{self.knockout} predict]  building V_pos at "
                  f"sigma_y_kernel={self._sigma_y_kernel:.2e} ...")
        V_pos, K_y_colsum = _build_V_pos_and_colsum_stabilised(
            self._train_score_csr, train_l_t, test_l_t,
            self._sigma_y_kernel, self.kernel_batch,
        )

        return _route_posterior(
            test_knn_idx_t, test_knn_sqd_t,
            V_pos, K_y_colsum,
            self.sigma_x_, self.batch_size,
        )
