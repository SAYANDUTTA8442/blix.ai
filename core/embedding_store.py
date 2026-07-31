"""
EmbeddingStore — persists and queries dense embedding vectors.  v0.2

Architecture
------------
Embeddings are stored as a NumPy ``.npy`` file (one row per memory entry)
alongside ``embedding_ids.json`` mapping matrix row → entry id.

Embedding model priority
------------------------
1. ``sentence-transformers`` (best quality, downloads once ~22 MB, cached).
2. TF-IDF + cosine similarity (offline fallback, zero dependencies beyond
   scikit-learn which is already a transitive dep).

TF-IDF consistency guarantee
------------------------------
Because the TF-IDF vocabulary is determined by the corpus, adding a single
document at a time changes the dimension on every call.  To prevent this:

* New texts are queued in ``_pending``.
* When ``_flush_pending()`` is called (after every ``add``), the vectorizer
  is re-fitted on ALL texts (corpus + pending) and the ENTIRE matrix is
  re-encoded from scratch to keep dimensions consistent.
* ``search()`` also triggers a flush so queries always reflect the latest
  vocabulary.

This makes writes O(n) in corpus size, which is fine for the expected
scale (<10 k memories).  The ``sentence-transformers`` path is O(1) per
add and is not affected.

Python 3.10 compatible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Any

import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)

_SBERT_WARNED = False
_TFIDF_MIN_DOCS = 2


class EmbeddingStore:
    """
    Manages a persistent matrix of sentence embeddings + id index.

    Parameters
    ----------
    embed_model_name:
        ``sentence-transformers`` model id (e.g. ``"all-MiniLM-L6-v2"``).
    embeddings_file:
        Path where the ``.npy`` matrix is persisted.
    ids_file:
        Path where the id↔row mapping JSON is persisted.
    threshold:
        Minimum cosine similarity (0–1) to qualify as a search hit.
    top_k:
        Maximum results returned per ``search()`` call.
    """

    def __init__(
        self,
        embed_model_name: str,
        embeddings_file: Path,
        ids_file: Path,
        threshold: float = 0.35,
        top_k: int = 5,
    ) -> None:
        self._model_name = embed_model_name
        self._embeddings_file = embeddings_file
        self._ids_file = ids_file
        self._threshold = threshold
        self._top_k = top_k

        # Lazy-loaded SBERT model
        self._sbert: Optional[Any] = None

        # TF-IDF state
        #   _corpus      : every text ever added (grows monotonically)
        #   _corpus_ids  : parallel list of entry ids
        #   _pending_*   : texts added since last full re-encode
        self._corpus: list[str] = []
        self._corpus_ids: list[int] = []
        self._pending_ids: list[int] = []
        self._pending_texts: list[str] = []

        # Committed in-memory matrix and id list (always consistent)
        self._matrix: Optional[np.ndarray] = None
        self._ids: list[int] = []

        self._load_index()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_index(self) -> None:
        if self._embeddings_file.exists() and self._ids_file.exists():
            try:
                self._matrix = np.load(str(self._embeddings_file))
                with self._ids_file.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                self._ids = data.get("ids", [])
                self._corpus = data.get("corpus", [])
                self._corpus_ids = data.get("corpus_ids", [])
                log.info("EmbeddingStore loaded %d vectors.", len(self._ids))
            except Exception as exc:
                log.warning("Cannot load index (%s) — starting empty.", exc)
                self._matrix = None
                self._ids = []
                self._corpus = []
                self._corpus_ids = []
        else:
            log.info("No embedding index found; starting empty.")

    def _save_index(self) -> None:
        if self._matrix is None or not self._ids:
            return
        self._embeddings_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(self._embeddings_file), self._matrix)
        payload = {
            "ids": self._ids,
            "corpus": self._corpus,
            "corpus_ids": self._corpus_ids,
        }
        with self._ids_file.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        log.debug("Saved %d embedding vectors.", len(self._ids))

    # ------------------------------------------------------------------
    # SBERT
    # ------------------------------------------------------------------

    def _get_sbert(self) -> Optional[Any]:
        if self._sbert is not None:
            return self._sbert
        try:
            import os
            from sentence_transformers import SentenceTransformer
            token = os.getenv("HF_TOKEN") or None
            kw: dict = {"token": token} if token else {}
            self._sbert = SentenceTransformer(self._model_name, **kw)
            log.info("SBERT model loaded: %s", self._model_name)
            return self._sbert
        except Exception as exc:
            global _SBERT_WARNED
            if not _SBERT_WARNED:
                log.warning("SBERT unavailable (%s). Using TF-IDF fallback.", exc)
                _SBERT_WARNED = True
            return None

    def _sbert_encode(self, texts: list[str]) -> Optional[np.ndarray]:
        model = self._get_sbert()
        if model is None:
            return None
        try:
            vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return vecs.astype(np.float32)
        except Exception as exc:
            log.warning("SBERT encode error: %s", exc)
            return None

    # ------------------------------------------------------------------
    # TF-IDF
    # ------------------------------------------------------------------

    def _tfidf_encode_all(self, texts: list[str]) -> Optional[np.ndarray]:
        """
        Fit a fresh TF-IDF vectorizer on *texts* and return the matrix.

        All texts are encoded together so every row has the same dimension.
        Returns None if fewer than _TFIDF_MIN_DOCS texts are provided.
        """
        if len(texts) < _TFIDF_MIN_DOCS:
            return None
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
            dense = vec.fit_transform(texts).toarray().astype(np.float32)
            norms = np.linalg.norm(dense, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return dense / norms
        except Exception as exc:
            log.warning("TF-IDF encode failed: %s", exc)
            return None

    def _tfidf_encode_query(self, query: str) -> Optional[np.ndarray]:
        """
        Encode a single *query* vector that is compatible with the current
        matrix.  Re-fits on corpus + query so the vocabulary matches.
        """
        all_texts = self._corpus + [query]
        if len(all_texts) < _TFIDF_MIN_DOCS:
            return None
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
            matrix = vec.fit_transform(all_texts).toarray().astype(np.float32)
            # Last row is the query; corpus rows may differ in dim from stored matrix
            # → we only return the query vector to be compared against a freshly
            #   re-encoded matrix (done in search())
            q_vec = matrix[-1:, :]
            norms = np.linalg.norm(q_vec, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return q_vec / norms
        except Exception as exc:
            log.warning("TF-IDF query encode failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Pending flush (TF-IDF mode)
    # ------------------------------------------------------------------

    def _flush_pending(self) -> None:
        """
        Integrate pending entries into the committed matrix.

        In SBERT mode: each entry is encoded independently — no flush needed
        (SBERT returns consistent 384-d vectors regardless of corpus).

        In TF-IDF mode: re-fit on the full corpus (committed + pending) and
        re-encode everything from scratch so dimensions stay consistent.
        """
        if not self._pending_ids:
            return

        if self._get_sbert() is not None:
            # SBERT path: just encode the pending batch and append
            vecs = self._sbert_encode(self._pending_texts)
            if vecs is not None:
                self._matrix = (
                    np.vstack([self._matrix, vecs])
                    if self._matrix is not None
                    else vecs
                )
                self._ids.extend(self._pending_ids)
                self._corpus.extend(self._pending_texts)
                self._corpus_ids.extend(self._pending_ids)
                self._pending_ids.clear()
                self._pending_texts.clear()
                self._save_index()
            return

        # TF-IDF path: re-encode ALL committed + pending texts
        all_ids = self._corpus_ids + self._pending_ids
        all_texts = self._corpus + self._pending_texts

        if len(all_texts) < _TFIDF_MIN_DOCS:
            # Not enough docs yet — keep in pending
            return

        vecs = self._tfidf_encode_all(all_texts)
        if vecs is None:
            return

        self._matrix = vecs
        self._ids = list(all_ids)
        self._corpus = list(all_texts)
        self._corpus_ids = list(all_ids)
        self._pending_ids.clear()
        self._pending_texts.clear()
        self._save_index()
        log.debug("Flushed %d total vectors.", len(self._ids))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, entry_id: int, text: str) -> Optional[int]:
        """
        Add *text* for *entry_id* to the index.

        Returns the row index in the matrix, or ``None`` when the text
        cannot be encoded yet (e.g. corpus too small for TF-IDF; the
        text is queued and will be encoded when the next ``add()`` or
        ``search()`` call provides enough corpus).
        """
        # Queue
        self._pending_ids.append(entry_id)
        self._pending_texts.append(text)

        # Try to flush
        pre = len(self._ids)
        self._flush_pending()
        post = len(self._ids)

        if post > pre:
            # This entry made it in
            row_idx = self._ids.index(entry_id) if entry_id in self._ids else post - 1
            log.debug("EmbeddingStore: added entry id=%d at row=%d", entry_id, row_idx)
            return row_idx

        log.debug("EmbeddingStore: entry id=%d queued (corpus size=%d)", entry_id, len(self._corpus) + len(self._pending_texts))
        return None

    def remove(self, entry_id: int) -> bool:
        """Remove the vector for *entry_id*. Returns True if removed."""
        if entry_id not in self._ids:
            return False
        row = self._ids.index(entry_id)
        assert self._matrix is not None
        self._matrix = np.delete(self._matrix, row, axis=0)
        if self._matrix.shape[0] == 0:
            self._matrix = None
        self._ids.pop(row)
        # Also remove from corpus
        if entry_id in self._corpus_ids:
            ci = self._corpus_ids.index(entry_id)
            self._corpus.pop(ci)
            self._corpus_ids.pop(ci)
        self._save_index()
        log.debug("Removed entry id=%d", entry_id)
        return True

    def search(self, query: str, top_k: Optional[int] = None) -> list[tuple[int, float]]:
        """
        Return ``(entry_id, cosine_similarity)`` pairs for the top-k most
        similar entries, filtered by threshold.
        """
        # Flush any pending so the matrix is up to date
        self._flush_pending()

        k = top_k if top_k is not None else self._top_k
        if self._matrix is None or not self._ids:
            return []

        # Encode the query
        if self._get_sbert() is not None:
            q_vec = self._sbert_encode([query])
        else:
            # TF-IDF: re-encode matrix + query together for dim consistency
            all_texts = self._corpus + [query]
            if len(all_texts) < _TFIDF_MIN_DOCS:
                return []
            vecs = self._tfidf_encode_all(all_texts)
            if vecs is None:
                return []
            # Last row is query; earlier rows = new encoding of corpus
            new_matrix = vecs[:-1]
            q_vec = vecs[-1:]
            # Update stored matrix to match new encoding
            self._matrix = new_matrix

        if q_vec is None:
            return []
        if q_vec.shape[1] != self._matrix.shape[1]:
            log.warning("Dimension mismatch in search.")
            return []

        sims: np.ndarray = (self._matrix @ q_vec.T).squeeze()
        if sims.ndim == 0:
            sims = np.array([float(sims)])

        hits = [
            (self._ids[i], float(sims[i]))
            for i in range(len(self._ids))
            if float(sims[i]) >= self._threshold
        ]
        hits.sort(key=lambda t: t[1], reverse=True)
        results = hits[:k]
        log.info(
            "EmbeddingStore.search(%r): %d hits (threshold=%.2f)",
            query[:40], len(results), self._threshold,
        )
        return results

    def rebuild(self, id_text_pairs: list[tuple[int, str]]) -> None:
        """Rebuild the entire index from scratch."""
        self._pending_ids.clear()
        self._pending_texts.clear()

        if not id_text_pairs:
            self._matrix = None
            self._ids = []
            self._corpus = []
            self._corpus_ids = []
            self._save_index()
            return

        log.info("EmbeddingStore.rebuild: %d entries…", len(id_text_pairs))
        ids_list = [p[0] for p in id_text_pairs]
        texts = [p[1] for p in id_text_pairs]

        vecs = self._sbert_encode(texts)
        if vecs is None:
            vecs = self._tfidf_encode_all(texts)
        if vecs is None:
            log.warning("rebuild: encoding failed.")
            return

        self._matrix = vecs
        self._ids = list(ids_list)
        self._corpus = list(texts)
        self._corpus_ids = list(ids_list)
        self._save_index()
        log.info("EmbeddingStore.rebuild: done (%d vectors).", len(self._ids))

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Committed entries in the index (excludes pending)."""
        return len(self._ids) + len(self._pending_ids)

    @property
    def indexed_ids(self) -> list[int]:
        """All entry ids — committed and pending."""
        return self._ids + self._pending_ids
