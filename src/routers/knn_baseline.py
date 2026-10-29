"""
kNN collaborative-filtering baseline for LLM routing.

Predicts s(q, l) as the mean of S_train[N_q(q), N_l(l)] — the average score
over the k_q nearest train queries and k_l nearest train LLMs.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


class KNNRouter:
    """User-item kNN baseline that operates in both query and model embedding spaces.

    Parameters
    ----------
    k_q : int  — number of nearest train queries to retrieve per test query.
    k_l : int  — number of nearest train LLMs to retrieve per test LLM.
    metric : str  — distance metric for NearestNeighbors (default: cosine).
    """

    def __init__(self, k_q: int = 10, k_l: int = 10, metric: str = "cosine"):
        self.k_q = k_q
        self.k_l = k_l
        self.metric = metric
        self._nn_q: NearestNeighbors | None = None
        self._nn_l: NearestNeighbors | None = None
        self._S: np.ndarray | None = None  # (Nq_train, Nl_train) float32

    def fit(
        self,
        train_q_emb: np.ndarray,   # (Nq_train, 768)  float32
        train_l_emb: np.ndarray,   # (Nl_train, 2048) float32
        S_train: np.ndarray,       # (Nq_train, Nl_train) uint8
    ) -> "KNNRouter":
        self._S = S_train.astype(np.float32)
        self._nn_q = NearestNeighbors(
            n_neighbors=self.k_q, metric=self.metric, algorithm="brute", n_jobs=-1
        ).fit(train_q_emb)
        self._nn_l = NearestNeighbors(
            n_neighbors=self.k_l, metric=self.metric, algorithm="brute", n_jobs=-1
        ).fit(train_l_emb)
        return self

    def predict(
        self,
        test_q_emb: np.ndarray,   # (Nq_test, 768)  float32
        test_l_emb: np.ndarray,   # (Nl_test, 2048) float32
    ) -> np.ndarray:              # (Nq_test, Nl_test) float32 — predicted scores in [0,1]
        if self._S is None:
            raise RuntimeError("Call fit() before predict().")

        # (Nq_test, k_q)  — indices into train query rows
        _, q_nbrs = self._nn_q.kneighbors(test_q_emb)
        # (Nl_test, k_l)  — indices into train LLM rows
        _, l_nbrs = self._nn_l.kneighbors(test_l_emb)

        Nq_test = test_q_emb.shape[0]
        Nl_test = test_l_emb.shape[0]
        S_hat = np.empty((Nq_test, Nl_test), dtype=np.float32)

        # For each test query, extract its k_q × k_l sub-block and take the mean.
        # We vectorize over test LLMs using advanced indexing.
        for i, qn in enumerate(q_nbrs):           # qn: (k_q,)
            # S_block: (k_q, Nl_test) — for each train-query neighbor, scores of all test-LLM neighborhoods
            # l_nbrs: (Nl_test, k_l); take mean over k_l dim first → (Nl_test,)
            block = self._S[np.ix_(qn, np.arange(self._S.shape[1]))]  # (k_q, Nl_train)
            # l_nbrs[j]: k_l indices into Nl_train → block[:, l_nbrs[j]].mean() = sub-block mean
            # Vectorized: block[:, l_nbrs] → (k_q, Nl_test, k_l); mean over axes 0 and 2
            S_hat[i] = block[:, l_nbrs].mean(axis=(0, 2))  # (Nl_test,)

        return S_hat


def predict_from_neighbors(
    S_train: np.ndarray,   # (Nq_train, Nl_train) float32
    q_nbrs: np.ndarray,    # (Nq_test, k_q) int — indices into train queries
    l_nbrs: np.ndarray,    # (Nl_test, k_l) int — indices into train LLMs
    chunk: int = 32,
) -> np.ndarray:           # (Nq_test, Nl_test) float32
    """Predict scores from precomputed neighbor index arrays.

    ŝ(q, l) = mean over the (k_q × k_l) sub-block of S_train at rows N_q(q)
    and cols N_l(l).  Vectorised over the test-query axis: accumulates a
    (Nq_test, Nl_train) sum over q-neighbors, then a (Nq_test, Nl_test) sum
    over l-neighbors.  The k axis is walked in chunks (default 32) so the
    intermediate fancy-indexed tensor stays bounded regardless of k_q × k_l.
    """
    Nq_train, Nl_train = S_train.shape
    Nq_test = q_nbrs.shape[0]
    Nl_test = l_nbrs.shape[0]
    k_q = q_nbrs.shape[1]
    k_l = l_nbrs.shape[1]

    # Fast path: k_q = all train queries AND k_l = all train LLMs → global mean.
    if k_q == Nq_train and k_l == Nl_train:
        return np.full((Nq_test, Nl_test), S_train.mean(), dtype=np.float32)

    # Fast path: k_q = all train queries → prediction collapses to a per-LLM mean
    # (same score for every test query).  Compute via col_mean to avoid allocating
    # the (Nq_train, Nl_test, k_l) intermediate tensor.
    if k_q == Nq_train:
        col_mean = S_train.mean(axis=0)          # (Nl_train,)
        col_scores = col_mean[l_nbrs].mean(axis=1)  # (Nl_test,)
        return np.broadcast_to(col_scores[None, :], (Nq_test, Nl_test)).copy()

    # Fast path: k_l = all train LLMs → prediction collapses to a per-query mean.
    if k_l == Nl_train:
        row_mean = S_train.mean(axis=1)          # (Nq_train,)
        row_scores = row_mean[q_nbrs].mean(axis=1)  # (Nq_test,)
        return np.broadcast_to(row_scores[:, None], (Nq_test, Nl_test)).copy()

    T = np.zeros((Nq_test, Nl_train), dtype=np.float32)
    for s in range(0, k_q, chunk):
        e = min(s + chunk, k_q)
        T += S_train[q_nbrs[:, s:e]].sum(axis=1)

    R = np.zeros((Nq_test, Nl_test), dtype=np.float32)
    for s in range(0, k_l, chunk):
        e = min(s + chunk, k_l)
        R += T[:, l_nbrs[:, s:e]].sum(axis=2)

    R /= float(k_q * k_l)
    return R


def predict_sweep(
    S_train: np.ndarray,   # (Nq_train, Nl_train) float32
    q_nbrs: np.ndarray,    # (Nq_test, k_q_max) int
    l_nbrs: np.ndarray,    # (Nl_test, k_l_max) int
    k_q_run,               # iterable of int — k_q values to evaluate
    k_l_run,               # iterable of int — k_l values to evaluate
    chunk: int = 32,
):
    """Yield (k_q, k_l, S_hat) for every (k_q, k_l) combination.

    Amortises work across the grid by maintaining a running (Nq_test, Nl_train)
    sum over q-neighbors and, for each k_q, a running (Nq_test, Nl_test) sum
    over l-neighbors. Total cost is
        O(Nq_test · max(k_q) · Nl_train)            (q-axis pass, once)
      + O(|k_q_run| · Nq_test · Nl_test · max(k_l)) (l-axis pass, once per k_q)
    instead of O(|grid| · Nq_test · k_q · Nl_test · k_l) for repeated calls.

    k values are deduplicated, capped at q_nbrs.shape[1] / l_nbrs.shape[1],
    and visited in ascending order. The yielded `S_hat` is a fresh array.
    """
    Nq_train, Nl_train = S_train.shape
    Nq_test = q_nbrs.shape[0]
    Nl_test = l_nbrs.shape[0]
    k_q_max = q_nbrs.shape[1]
    k_l_max = l_nbrs.shape[1]
    k_q_sorted = sorted({min(int(k), k_q_max) for k in k_q_run})
    k_l_sorted = sorted({min(int(k), k_l_max) for k in k_l_run})

    T = np.zeros((Nq_test, Nl_train), dtype=np.float32)
    prev_k_q = 0
    for k_q in k_q_sorted:
        for s in range(prev_k_q, k_q, chunk):
            e = min(s + chunk, k_q)
            T += S_train[q_nbrs[:, s:e]].sum(axis=1)
        prev_k_q = k_q

        R = np.zeros((Nq_test, Nl_test), dtype=np.float32)
        prev_k_l = 0
        for k_l in k_l_sorted:
            for s in range(prev_k_l, k_l, chunk):
                e = min(s + chunk, k_l)
                R += T[:, l_nbrs[:, s:e]].sum(axis=2)
            prev_k_l = k_l
            yield k_q, k_l, R / float(k_q * k_l)
