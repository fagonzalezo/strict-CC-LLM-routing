"""
Query-independent (static) routing policies, expressed in the project's router contract.

A static policy sends *every* query to the same LLM at a given budget cap. The whole point
of this module is that such a policy is a **rank-1 prediction matrix**: give
`src/metrics.py::per_query_cap_curve` an `S_hat` whose rows are all identical --
`S_hat[q, l] = w[l]` -- and its `argmax` over the affordable columns picks the same
highest-`w` model for every query. So a static policy is scored by the paper's own primary
protocol with no new metric code, and drops into `src/eval/runner.py::evaluate_predictions`
unchanged.

Two things this module exists to get right:

1. **The `[0, 1]` contract.** `predict()` must return probabilities, because
   `calibration_metrics` / `reliability_curve` run on whatever it returns. A raw preference
   vector (a parameter count, or a ridge prediction that can go slightly negative) would feed
   nonsense to them. Any *monotone* rescale preserves the argmax and therefore every
   cost-curve number, so `StaticRouter` min-max rescales `w` into `[0, 1]`. The calibration
   fields are still meaningless for a cost-derived `w` and callers null them out -- rescaling
   makes the numbers well-formed, not meaningful.

2. **Tie-breaking (`largest_affordable_curve`).** `np.argmax` silently takes the *first*
   maximal column, so a policy that ranks by size alone would score as an artifact of column
   order: 146 of RouterEval's 376 test LLMs are 7B, and 12 of EmbedLLM's 21. Jitter would
   only hide that. The one order-independent reading is to average over the tied set, which
   is what `largest_affordable_curve` computes -- the floor's random-affordable logic
   restricted to the top size stratum.
"""

from __future__ import annotations

import numpy as np


class StaticRouter:
    """A query-independent policy: score every LLM by `w`, ignore the query.

    Follows the project's sklearn-style contract -- `fit(...) -> self`,
    `predict(Q, L) -> (Nq, Nl)` in `[0, 1]` -- so it drops into
    `src/eval/runner.py::evaluate()` and the all-methods comparison machinery unchanged.
    `w` is supplied at construction time (fitted elsewhere, or not fitted at all), so `fit`
    is a no-op that accepts either the 3-arg or the 6-arg signature.
    """

    def __init__(self, w: np.ndarray, *, name: str = "static"):
        w = np.asarray(w, dtype=np.float64).ravel()
        if w.size == 0:
            raise ValueError("w must be non-empty")
        if not np.all(np.isfinite(w)):
            raise ValueError("w must be finite")
        self.w = w
        self.name = name
        # Monotone min-max into [0, 1]: preserves the argmax, hence every cost-curve number.
        span = float(w.max() - w.min())
        self.w_scaled_ = np.full_like(w, 0.5) if span == 0.0 else (w - w.min()) / span

    def fit(self, *args, **kwargs) -> "StaticRouter":
        return self

    def predict(self, test_q_emb, test_l_emb) -> np.ndarray:
        n_q = np.asarray(test_q_emb).shape[0]
        n_l = np.asarray(test_l_emb).shape[0]
        if n_l != self.w.size:
            raise ValueError(
                f"{self.name}: w has {self.w.size} entries but predict() got {n_l} LLMs"
            )
        return np.broadcast_to(self.w_scaled_.astype(np.float32), (n_q, n_l)).copy()


def rank1_predictions(w: np.ndarray, n_queries: int) -> np.ndarray:
    """`w` broadcast to `(n_queries, len(w))`, rescaled into `[0, 1]`.

    The direct form of what `StaticRouter.predict` returns, for callers that already have
    the test-split shapes and do not need a router object.
    """
    return StaticRouter(w).predict(np.empty((n_queries, 0)), np.empty((len(np.asarray(w).ravel()), 0)))


def largest_affordable_curve(S_true: np.ndarray, c: np.ndarray) -> list[tuple[float, float]]:
    """S1's cap curve: at each cap, the *tied-largest* affordable models, averaged.

    Mirrors `src/metrics.py::per_query_cap_curve`'s grid exactly -- one x-point per distinct
    LLM cost, `np.unique(c)` -- so the series shares the x-grid every other curve in this
    project uses and `src/eval/plots.py::_assert_shared_grid` passes.

    At cap `b` the affordable set is `A = {l : c[l] <= b}` and the top stratum is
    `T = {l in A : c[l] == max c over A}`. Quality is `S_true[:, T].mean()`. When `|T| == 1`
    this is exactly "route everything to the largest affordable model"; when several models
    tie at that size it is the expectation over a uniform choice among them, which is the
    only reading that does not depend on column order.
    """
    S_true = np.asarray(S_true, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    if S_true.shape[1] != c.shape[0]:
        raise ValueError(f"S_true has {S_true.shape[1]} columns but c has {c.shape[0]} entries")
    out = []
    for b in np.unique(c):
        affordable = np.flatnonzero(c <= b)
        top = affordable[c[affordable] == c[affordable].max()]
        out.append((float(b), float(S_true[:, top].mean())))
    return out


def largest_affordable_per_query(S_true: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Per-query realised quality of S1 at each cap: `(n_queries, n_caps)`.

    `largest_affordable_curve` is this matrix's column mean. Bootstrapping resamples query
    *rows*, so the paired bootstrap needs the per-query decomposition rather than the
    already-pooled curve.
    """
    S_true = np.asarray(S_true, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    caps = np.unique(c)
    out = np.empty((S_true.shape[0], caps.size), dtype=np.float64)
    for j, b in enumerate(caps):
        affordable = np.flatnonzero(c <= b)
        top = affordable[c[affordable] == c[affordable].max()]
        out[:, j] = S_true[:, top].mean(axis=1)
    return out
