"""
Shared evaluation suite for LLM-routing experiments (exp_18 onward).

Implements the "beyond top-1" protocol from docs/routing_eval_beyond_top1.md:

  A. Ranking metrics   (cost-free): top1, top1_regret, pq_auc, recall@k,
                                    ndcg@k, mrr, llm_ranking_correlation
  B. Calibration       (cost-free): brier_score (== mse), ece, reliability_curve
  C. Cost-aware        (post-hoc λ-sweep; cost NOT used in training):
                                    cost_quality_curve, curve_auc, aiq, apgr,
                                    quality_at_budget, cost_at_target_quality,
                                    single_llm_points, random_mixture_hull,
                                    random_within_budget_floor, oracle_curve

Convention everywhere:
    S_hat  : (Nq, Nl) float — predicted score q̂(query, llm) in [0, 1]
    S_true : (Nq, Nl) {0,1} — binary correctness label
A "query" is a row; an "LLM" is a column. Ranking metrics that need a defined
answer skip degenerate queries (no correct LLM, or all-correct), mirroring the
per-query-AUC convention used since exp_11.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr

# np.trapz was renamed np.trapezoid in NumPy 2.0; support both.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

# ===========================================================================
# A. Ranking metrics (cost-free)
# ===========================================================================


def top1_accuracy(S_hat: np.ndarray, S_true: np.ndarray) -> float:
    """Mean correctness of the argmax-selected LLM."""
    best = np.argmax(S_hat, axis=1)
    return float(S_true[np.arange(len(best)), best].mean())


def top1_regret(S_hat: np.ndarray, S_true: np.ndarray) -> float:
    """Oracle correctness − selected correctness (averaged over queries)."""
    best_pred = np.argmax(S_hat, axis=1)
    selected = S_true[np.arange(len(best_pred)), best_pred].astype(float)
    oracle = S_true.max(axis=1).astype(float)
    return float((oracle - selected).mean())


def per_query_auc(S_hat: np.ndarray, S_true: np.ndarray) -> float:
    """Mean per-query ROC AUC; skips degenerate rows (all-correct / all-wrong).

    Probability a correct LLM is ranked above an incorrect one. This is the
    primary selection metric for exp_18 (`pq_auc`).

    Vectorized via the Mann-Whitney U / rank-sum identity (equivalent to
    sklearn's per-row roc_auc_score with average tie-breaking): a Python loop
    calling sklearn once per row does not scale to bootstrap resampling over
    10k+ queries (build_comparison.py's H3 bootstrap timed out on this).
    """
    S_true = np.asarray(S_true)
    n_pos = S_true.sum(axis=1)
    n_neg = S_true.shape[1] - n_pos
    valid = (n_pos > 0) & (n_neg > 0)
    if not valid.any():
        return float("nan")
    ranks = rankdata(np.asarray(S_hat)[valid], axis=1, method="average")
    sum_ranks_pos = (ranks * S_true[valid]).sum(axis=1)
    n_pos, n_neg = n_pos[valid], n_neg[valid]
    aucs = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(aucs.mean())


def recall_at_k(S_hat: np.ndarray, S_true: np.ndarray, k: int) -> float:
    """Fraction of *solvable* queries with ≥1 correct LLM in the predicted top-k.

    Conditioned on queries that have at least one correct LLM (a query with no
    correct LLM can never be a hit), consistent with the pq_auc degenerate-row
    convention.
    """
    Nq, Nl = S_hat.shape
    k = min(int(k), Nl)
    topk = np.argpartition(-S_hat, k - 1, axis=1)[:, :k]
    rows = np.arange(Nq)[:, None]
    hit = S_true[rows, topk].max(axis=1)  # 1 if any correct LLM in top-k
    solvable = S_true.max(axis=1) > 0
    if not solvable.any():
        return float("nan")
    return float(hit[solvable].mean())


def ndcg_at_k(S_hat: np.ndarray, S_true: np.ndarray, k: int) -> float:
    """Mean NDCG@k with binary gains over the predicted ordering.

    Averaged over solvable queries; IDCG uses min(#correct, k) ideal positions.
    """
    Nq, Nl = S_hat.shape
    k = min(int(k), Nl)
    order = np.argsort(-S_hat, axis=1)[:, :k]
    rows = np.arange(Nq)[:, None]
    gains = S_true[rows, order].astype(float)  # (Nq, k) in ranked order
    discounts = 1.0 / np.log2(np.arange(2, k + 2))  # (k,)
    dcg = (gains * discounts).sum(axis=1)
    n_correct = S_true.sum(axis=1).astype(int)

    ndcgs = []
    for i in range(Nq):
        nc = min(int(n_correct[i]), k)
        if nc == 0:
            continue
        idcg = discounts[:nc].sum()
        ndcgs.append(dcg[i] / idcg)
    return float(np.mean(ndcgs)) if ndcgs else float("nan")


def mrr(S_hat: np.ndarray, S_true: np.ndarray) -> float:
    """Mean reciprocal rank of the first correct LLM in the predicted order."""
    Nq = S_hat.shape[0]
    order = np.argsort(-S_hat, axis=1)
    rows = np.arange(Nq)[:, None]
    ranked_rel = S_true[rows, order]  # (Nq, Nl) binary in ranked order
    has = ranked_rel.max(axis=1) > 0
    first = ranked_rel.argmax(axis=1)  # index of first 1 (0 if all-zero — masked out)
    if not has.any():
        return float("nan")
    rr = 1.0 / (first[has] + 1.0)
    return float(rr.mean())


def llm_ranking_correlation(S_hat: np.ndarray, S_true: np.ndarray) -> dict:
    """Spearman ρ and Kendall τ between per-LLM mean predicted score and per-LLM
    true accuracy rate — does the router recover the *global* LLM quality order?"""
    pred_rate = S_hat.mean(axis=0)
    true_rate = S_true.mean(axis=0)
    # Undefined when either ranking is constant (e.g. the global-mean baseline,
    # which scores every LLM identically).
    if pred_rate.min() == pred_rate.max() or true_rate.min() == true_rate.max():
        return {"spearman": float("nan"), "kendall": float("nan")}
    sp = spearmanr(pred_rate, true_rate).correlation
    kt = kendalltau(pred_rate, true_rate).correlation
    return {"spearman": float(sp), "kendall": float(kt)}


# ===========================================================================
# B. Calibration metrics (cost-free)
# ===========================================================================


def brier_score(S_hat: np.ndarray, S_true: np.ndarray) -> float:
    """Mean squared error of q̂ vs binary correctness (proper scoring rule)."""
    return float(((S_hat - S_true.astype(np.float32)) ** 2).mean())


# Backwards-compatible alias (exp_1–17 called this `mse`).
mse = brier_score


def expected_calibration_error(
    S_hat: np.ndarray, S_true: np.ndarray, n_bins: int = 10
) -> float:
    """Expected Calibration Error over equal-width confidence bins (flattened)."""
    conf = S_hat.ravel().astype(float)
    acc = S_true.ravel().astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    N = conf.size
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += (m.sum() / N) * abs(conf[m].mean() - acc[m].mean())
    return float(ece)


def reliability_curve(
    S_hat: np.ndarray, S_true: np.ndarray, n_bins: int = 10
) -> dict:
    """Per-bin confidence, empirical accuracy, and count — for reliability diagrams."""
    conf = S_hat.ravel().astype(float)
    acc = S_true.ravel().astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    bin_conf, bin_acc, bin_count = [], [], []
    for b in range(n_bins):
        m = idx == b
        bin_count.append(int(m.sum()))
        bin_conf.append(float(conf[m].mean()) if m.any() else float("nan"))
        bin_acc.append(float(acc[m].mean()) if m.any() else float("nan"))
    return {
        "bin_edges": edges.tolist(),
        "bin_conf": bin_conf,
        "bin_acc": bin_acc,
        "bin_count": bin_count,
    }


# ===========================================================================
# C. Cost-aware metrics (post-hoc; cost never enters training)
#
# PRIMARY path — per-query budget-cap (exp_18 onward):
#   per_query_cap_curve, cap_router_envelope, cap_curve_auc, cap_aiq,
#   cap_quality_at_budget, cap_cost_at_target_quality
#   cost_metrics()  ← calls this path
#
# LEGACY / overlay — average-budget Lagrangian sweep (exp_17 and earlier):
#   default_lambda_grid, route_utility, cost_quality_curve, pareto_front,
#   curve_auc, random_mixture_hull, oracle_curve, aiq
#   Kept for back-compat and optional overlay; removed from cost_metrics().
# ===========================================================================


def default_lambda_grid(c: np.ndarray, n: int = 50) -> np.ndarray:
    """A λ grid spanning quality-only (0) to cost-only (cheapest dominates).

    λ multiplies cost; selection flips once λ·Δcost ≈ Δq̂ (≈1). The upper end is
    chosen so the cheapest LLM wins regardless of score, plus an explicit np.inf
    sentinel handled by route_utility.
    """
    c = np.asarray(c, dtype=float)
    spread = max(c.max() - c.min(), 1e-6)
    hi = 10.0 / spread  # well past where cost dominates a ≤1 score gap
    grid = np.concatenate([[0.0], np.logspace(-4, np.log10(hi), n - 1), [np.inf]])
    return grid


def route_utility(S_hat: np.ndarray, c: np.ndarray, lam: float) -> np.ndarray:
    """Per-query selected LLM index under utility q̂ − λ·c.

    Utility ties are broken toward **lower cost**. This matters at the
    quality-only end (λ=0) on binary labels, where many LLMs share the top
    score: a plain argmax would return an arbitrary lowest-index winner and
    inflate the x-coordinate, whereas the cheapest top-utility LLM is the
    honest cost of that routing decision.
    """
    c = np.asarray(c, dtype=float)
    if np.isinf(lam):
        return np.full(S_hat.shape[0], int(np.argmin(c)), dtype=int)
    util = S_hat - lam * c[None, :]
    best = util.max(axis=1, keepdims=True)
    eligible = util >= best - 1e-12
    masked_cost = np.where(eligible, c[None, :], np.inf)
    return masked_cost.argmin(axis=1)


def cost_quality_curve(
    S_hat: np.ndarray, S_true: np.ndarray, c: np.ndarray, lambdas=None
) -> list[tuple[float, float]]:
    """(avg_cost, avg_quality) points, one per λ. λ=0 → quality-only argmax."""
    c = np.asarray(c, dtype=float)
    if lambdas is None:
        lambdas = default_lambda_grid(c)
    rows = np.arange(S_hat.shape[0])
    pts = []
    for lam in lambdas:
        sel = route_utility(S_hat, c, lam)
        pts.append((float(c[sel].mean()), float(S_true[rows, sel].mean())))
    return pts


def _cross(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def pareto_front(points) -> list[tuple[float, float]]:
    """Upper-left staircase: cheapest-first, keep points of strictly rising quality."""
    pts = sorted(set((float(c), float(q)) for c, q in points))
    front, best_q = [], -np.inf
    for c_, q_ in pts:
        if q_ > best_q:
            front.append((c_, q_))
            best_q = q_
    return front


def curve_auc(points, normalize: bool = True) -> float:
    """Trapezoidal area of quality vs cost over the Pareto front.

    Normalized by the cost span → mean achievable quality per unit cost range,
    the headline cost scalar. Falls back to the single quality value when the
    front collapses to one cost.
    """
    front = pareto_front(points)
    if not front:
        return float("nan")
    if len(front) == 1:
        return float(front[0][1])
    cs = np.array([p[0] for p in front])
    qs = np.array([p[1] for p in front])
    area = float(_trapz(qs, cs))
    if normalize:
        rng = cs[-1] - cs[0]
        return area / rng if rng > 0 else float(qs.mean())
    return area


def single_llm_points(S_true: np.ndarray, c: np.ndarray) -> list[tuple[float, float]]:
    """One (cost, mean-quality) dot per LLM."""
    q = S_true.mean(axis=0)
    c = np.asarray(c, dtype=float)
    return [(float(c[m]), float(q[m])) for m in range(len(c))]


def random_mixture_hull(points) -> list[tuple[float, float]]:
    """Upper convex hull of a point cloud — the zero-skill random-mixture baseline.

    Randomly mixing two LLMs traces the straight line between their points, so
    the achievable zero-skill region is the upper convex hull. A router adds
    value only if its curve lies above this hull.
    """
    pts = sorted(set((float(c), float(q)) for c, q in points))
    if len(pts) <= 2:
        return pts
    upper = []
    for p in reversed(pts):  # right-to-left build → reverse for ascending cost
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return list(reversed(upper))


def random_within_budget_floor(
    S_true: np.ndarray, c: np.ndarray
) -> list[tuple[float, float]]:
    """Deployable zero-skill cost floor for cold-cold routing.

    Knows only per-LLM cost (public metadata), never per-LLM quality. For each
    budget threshold b (each distinct cost) it routes uniformly at random among
    the LLMs with cost ≤ b, so the realised point is the mean (cost, true
    quality) of that affordable set. Sweeping b traces the cheapest-first prefix
    curve — the honest floor a real router must beat, unlike the LLM-marginal
    oracle (random_mixture_hull), which needs the held-out test-quality labels.
    """
    c = np.asarray(c, dtype=float)
    q = np.asarray(S_true, dtype=float).mean(axis=0)
    pts = []
    for b in np.unique(c):
        m = c <= b
        pts.append((float(c[m].mean()), float(q[m].mean())))
    return pts


def oracle_curve(S_true: np.ndarray, c: np.ndarray) -> list[tuple[float, float]]:
    """Per-query cost-quality upper bound: each query routed to its cheapest
    CORRECT LLM, budget swept by solving cheapest-to-solve queries first.

    Unsolvable queries (no correct LLM) sit at the globally cheapest LLM
    (quality 0). The frontier rises from the cheapest LLM's own accuracy up to
    P(solvable). Computed directly — not via a λ-sweep — because with binary
    labels the score gap is exactly 1, so a λ grid tuned for continuous scores
    (hi ≈ 0.12) never exceeds 1/Δcost and the sweep collapses to ≈2 points
    with wrong x-coordinates (verified empirically: gsm8k oracle_auc = 0.500
    with the old sweep, vs the correct P(solvable) ≈ 0.97).
    """
    c = np.asarray(c, dtype=float)
    S = np.asarray(S_true)
    Nq = S.shape[0]
    c_min = float(c.min())
    # Per-query cheapest correct LLM cost; inf for unsolvable queries.
    cc = np.where(S > 0, c[None, :], np.inf).min(axis=1)
    cc_solv = np.sort(cc[np.isfinite(cc)])   # solvable queries, ascending cost
    # Build frontier: start with all queries at the cheapest LLM (quality 0),
    # then "upgrade" queries in cheapest-to-solve order.
    base = c_min * Nq                        # total cost at baseline
    pts: list[tuple[float, float]] = []
    running = 0.0
    for k in range(1, cc_solv.size + 1):
        running += cc_solv[k - 1] - c_min   # extra cost vs cheapest LLM
        pts.append(((base + running) / Nq, k / Nq))
    return pts if pts else [(c_min, 0.0)]


def aiq(router_points, baseline_points) -> float:
    """Area between the router curve and a baseline curve (RouterBench AIQ
    analogue), as the difference of normalized curve AUCs. The headline AIQ uses
    the deployable cost-only floor as baseline; pass the LLM-marginal oracle hull
    to measure the gap to that (privileged) reference instead."""
    return curve_auc(router_points) - curve_auc(baseline_points)


def apgr(router_auc: float, weakest_q: float, strongest_q: float) -> float:
    """Performance-Gap-Recovered: curve AUC normalized to the weakest→strongest
    single-LLM quality gap."""
    denom = strongest_q - weakest_q
    return float((router_auc - weakest_q) / denom) if denom > 0 else float("nan")


def quality_at_budget(points, budget: float) -> float:
    """Interpolated router-curve quality at a given average cost."""
    front = pareto_front(points)
    if not front:
        return float("nan")
    cs = np.array([p[0] for p in front])
    qs = np.array([p[1] for p in front])
    return float(np.interp(budget, cs, qs))


def cost_at_target_quality(points, q_target: float) -> float:
    """Cheapest average cost on the router front reaching q_target (NaN if never)."""
    for c_, q_ in pareto_front(points):
        if q_ >= q_target:
            return float(c_)
    return float("nan")


# ===========================================================================
# C-primary. Per-query budget-cap curves (exp_18+)
# ===========================================================================


def per_query_cap_curve(
    S_hat: np.ndarray, S_true: np.ndarray, c: np.ndarray
) -> dict:
    """Per-query budget-cap curves: x = cap b, three series.

    For each distinct LLM cost b, every query independently may only use LLMs
    with cost ≤ b.  Sweeping b over all unique costs yields:

      router  = argmax q̂ within cap → realised correctness (may dip if
                the router's scores are miscalibrated at some cap values)
      floor   = random affordable LLM → mean S_true over all (query, affordable
                LLM) pairs; the deployable zero-skill baseline
      oracle  = max S_true within cap per query → P(solvable ≤ b); ceiling

    All three share the same x-axis [c_min, c_max].  No cross-query cost
    reallocation (barbell-free).  Returned as a dict of (b, quality) lists.
    """
    c = np.asarray(c, dtype=float)
    S_hat = np.asarray(S_hat, dtype=float)
    S_true = np.asarray(S_true, dtype=float)
    rows = np.arange(S_hat.shape[0])
    r_pts, f_pts, o_pts = [], [], []
    for b in np.unique(c):
        idx = np.where(c <= b)[0]
        sel = idx[S_hat[:, idx].argmax(axis=1)]              # router argmax
        r_pts.append((float(b), float(S_true[rows, sel].mean())))
        f_pts.append((float(b), float(S_true[:, idx].mean())))              # floor
        o_pts.append((float(b), float(S_true[:, idx].max(axis=1).mean())))  # oracle
    return {"router": r_pts, "floor": f_pts, "oracle": o_pts}


def cap_router_envelope(
    router_test: list[tuple[float, float]],
    router_val: list[tuple[float, float]],
    c: np.ndarray,
) -> list[tuple[float, float]]:
    """Val-tuned monotone envelope of the per-query-cap router curve.

    At each budget ceiling B (over sorted unique test costs c), picks the best
    inner cap b'* ≤ B by maximising router_val(b') on the validation set, then
    reads the corresponding test quality router_test(b'*).  Monotone by
    construction; deployable because no test labels are used.

    The gap between the raw router curve and this envelope is the miscalibration
    cost — budgets where the router would have done better by capping earlier.
    """
    c = np.asarray(c, dtype=float)
    bs = np.unique(c)
    v_cs = np.array([p[0] for p in router_val])
    v_qs = np.array([p[1] for p in router_val])
    t_cs = np.array([p[0] for p in router_test])
    t_qs = np.array([p[1] for p in router_test])
    env = []
    for B in bs:
        cands = bs[bs <= B]
        val_q = np.interp(cands, v_cs, v_qs, left=v_qs[0], right=v_qs[-1])
        b_star = float(cands[int(val_q.argmax())])
        q_te = float(np.interp(b_star, t_cs, t_qs, left=t_qs[0], right=t_qs[-1]))
        env.append((float(B), q_te))
    return env


def cap_curve_auc(
    curve: list[tuple[float, float]], lo: float, hi: float, log: bool = False
) -> float:
    """Mean quality of a cap curve over a COMMON support [lo, hi].

    Holds the curve flat beyond its first/last point (constant extrapolation).
    Use identical (lo, hi) for all curves so AUC values are directly comparable.
    Linear grid by default; log=True for a log-spaced robustness check.
    """
    if not curve:
        return float("nan")
    cs = np.array([p[0] for p in curve])
    qs = np.array([p[1] for p in curve])
    grid = np.geomspace(lo, hi, 3000) if log else np.linspace(lo, hi, 3000)
    return float(np.interp(grid, cs, qs, left=qs[0], right=qs[-1]).mean())


def cap_aiq(
    router_curve: list[tuple[float, float]],
    floor_curve: list[tuple[float, float]],
    lo: float,
    hi: float,
    log: bool = False,
) -> float:
    """AIQ as area-between-curves over a shared support [lo, hi].

    cap_curve_auc(router) − cap_curve_auc(floor).  Positive = router above floor.
    Both curves must be evaluated on the same (lo, hi) to be comparable.
    """
    return (
        cap_curve_auc(router_curve, lo, hi, log)
        - cap_curve_auc(floor_curve, lo, hi, log)
    )


def cap_quality_at_budget(curve: list[tuple[float, float]], b: float) -> float:
    """Quality of a cap curve at budget cap b (flat-tail interpolation)."""
    if not curve:
        return float("nan")
    cs = np.array([p[0] for p in curve])
    qs = np.array([p[1] for p in curve])
    return float(np.interp(b, cs, qs, left=qs[0], right=qs[-1]))


def cap_cost_at_target_quality(
    curve: list[tuple[float, float]], q_target: float
) -> float:
    """Minimum cap b at which the curve first reaches q_target (NaN if never)."""
    for b, q in sorted(curve):
        if q >= q_target:
            return float(b)
    return float("nan")


# ===========================================================================
# Convenience bundles
# ===========================================================================


def ranking_metrics(S_hat, S_true, ks=(1, 3, 5, 10)) -> dict:
    """Cost-free ranking suite (group A)."""
    out = {
        "top1_acc": top1_accuracy(S_hat, S_true),
        "top1_regret": top1_regret(S_hat, S_true),
        "pq_auc": per_query_auc(S_hat, S_true),
        "mrr": mrr(S_hat, S_true),
    }
    for k in ks:
        out[f"recall_at_{k}"] = recall_at_k(S_hat, S_true, k)
        out[f"ndcg_at_{k}"] = ndcg_at_k(S_hat, S_true, k)
    corr = llm_ranking_correlation(S_hat, S_true)
    out["llm_spearman"] = corr["spearman"]
    out["llm_kendall"] = corr["kendall"]
    return out


def calibration_metrics(S_hat, S_true, n_bins=10) -> dict:
    """Cost-free calibration suite (group B); reliability bins fetched separately."""
    return {
        "brier": brier_score(S_hat, S_true),
        "ece": expected_calibration_error(S_hat, S_true, n_bins),
    }


def cost_metrics(
    S_hat,
    S_true,
    c,
    deploy_window: float | None = None,
    cap_budgets: list | None = None,
    q_frac: float = 0.95,
) -> dict:
    """Cost-aware metrics (group C) + curves — per-query budget-cap protocol.

    Primary object: per_query_cap_curve (x = cap b).  All three series
    (router / floor / oracle) share the same support [c_min, c_max], so AUC
    and AIQ are well-posed same-support comparisons.

    deploy_window : upper cap for the deployment-window AIQ (default: median c)
    cap_budgets   : caps for quality@cap scalars (default: 1, 3, 7, 30, c_max)
    q_frac        : target quality as fraction of the strongest single LLM

    The curves dict also reserves a "router_envelope" slot (None here) that
    run_knn_eval.py fills with the val-tuned monotone envelope after computing
    the val S_hat.
    """
    c = np.asarray(c, dtype=float)
    c_min, c_max = float(c.min()), float(c.max())
    if deploy_window is None:
        deploy_window = float(np.median(c))
    if cap_budgets is None:
        cap_budgets = sorted({1.0, 3.0, 7.0, 30.0, c_max})

    cap_crvs = per_query_cap_curve(S_hat, S_true, c)
    single_pts = single_llm_points(S_true, c)
    single_q = np.array([q for _, q in single_pts])
    weakest_q = float(single_q.min())
    strongest_q = float(single_q.max())
    q_target = q_frac * strongest_q

    # AUC scalars over shared support (linear, full range)
    r_auc = cap_curve_auc(cap_crvs["router"], c_min, c_max)
    f_auc = cap_curve_auc(cap_crvs["floor"],  c_min, c_max)
    o_auc = cap_curve_auc(cap_crvs["oracle"], c_min, c_max)

    # AIQ: full range and deployment window
    aiq_full   = cap_aiq(cap_crvs["router"], cap_crvs["floor"], c_min, c_max)
    aiq_deploy = cap_aiq(cap_crvs["router"], cap_crvs["floor"], c_min, deploy_window)

    # Quality@cap for each budget cap and each series
    q_at = {}
    for b in cap_budgets:
        b_key = f"cap_{b:.0f}B"
        for k in ("router", "floor", "oracle"):
            q_at[f"quality_at_{b_key}_{k}"] = cap_quality_at_budget(cap_crvs[k], b)

    scalars = {
        # Full-range linear AUC (shared support — values are comparable)
        "cost_auc":        r_auc,
        "floor_cost_auc":  f_auc,
        "oracle_cost_auc": o_auc,
        # AIQ (true area-between-curves)
        "aiq":             aiq_full,
        "aiq_deploy":      aiq_deploy,
        # APGR and single-LLM anchors
        "apgr":            apgr(r_auc, weakest_q, strongest_q),
        "weakest_single_q":  weakest_q,
        "strongest_single_q": strongest_q,
        # Budget metadata
        "deploy_window": float(deploy_window),
        "q_target":      float(q_target),
        # Minimum cap to reach q_frac of the strongest single LLM
        "cost_at_target_quality": cap_cost_at_target_quality(
            cap_crvs["router"], q_target
        ),
        **q_at,
    }
    curves = {
        "router":          cap_crvs["router"],
        "floor":           cap_crvs["floor"],
        "oracle":          cap_crvs["oracle"],
        "single_llm":      single_pts,
        # Filled by run_knn_eval.py after val prediction:
        "router_envelope": None,
    }
    return {"scalars": scalars, "curves": curves}
