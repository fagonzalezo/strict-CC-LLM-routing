# Vendored ColdRec (AGNN, FSGNN)

The subset of [ColdRec](https://github.com/YuanchenBei/ColdRec), the cold-start
recommendation benchmark by Yuanchen Bei et al., that the AGNN and FSGNN baselines need. It is
distributed under the MIT license in [`LICENSE`](LICENSE). The code was copied from the fork
[`fagonzalezo/ColdRec`](https://github.com/fagonzalezo/ColdRec), branch `analysis`, commit
`7c6ceea7bdbe88029b23575fbbcaaa6cb251005f`. Modules the two models never import were left out.

ColdRec evaluates one cold side at a time (`cold_object` = `'user'` or `'item'`). Strict
Cold--Cold routing needs both sides cold, so four local patches add `cold_object='both'`:

| Patch | File | Change |
|---|---|---|
| A1 | `model/AGNN.py` `__init__` | require content features on **both** sides when `cold_object='both'` |
| A2 | `model/AGNN.py` `loss()` | activate both VAE branches (query and LLM); sum their reconstruction, KL and alignment losses |
| A3 | `model/AGNN.py` `train()`, `save()` | generate embeddings for cold users **and** cold items |
| F1 | `model/FSGNN.py` `_lfc_loss()` | restrict the feature-completion loss to the cold entities of both sides (otherwise it falls through to a loss over all, including warm, entities) |

The modules import each other by bare name (`util.*`, `model.*`), so the driver
`src/experiments/coldrec_baselines/run_coldrec.py` puts this directory on `sys.path`.
