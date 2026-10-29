"""Dataset-specific answer collection.

Each module here is consumed only by `src/data_pipeline/fingerprints/collect_answers.py` to
*populate* an `AnswerStore` — encoding always reads back through the store
(`AnswerStore.answers_for`), so `build_fingerprints.py` never imports anything from this
package. That is what keeps `src/fingerprints/encode.py` dataset-agnostic.
"""
