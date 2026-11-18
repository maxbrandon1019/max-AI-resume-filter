"""Hybrid Vector Store combining Dense Vector Search and Sparse BM25 with Metadata Filtering."""
import re
import math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from src.models.schemas import ChunkDocument, ResumeSectionType
from src.retrieval.embeddings import EmbeddingManager


class HybridVectorStore:
    """In-memory hybrid dense + sparse vector store with section weighting and metadata filtering."""

    def __init__(self, embedding_manager: Optional[EmbeddingManager] = None):
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.chunks: List[ChunkDocument] = []
        self.dense_embeddings: Optional[np.ndarray] = None
        self.corpus_tokens: List[List[str]] = []
        
        # Section importance multipliers
        self.section_weights = {
            ResumeSectionType.SKILLS.value: 1.3,
            ResumeSectionType.EXPERIENCE.value: 1.2,
            ResumeSectionType.PROJECTS.value: 1.1,
            ResumeSectionType.SUMMARY.value: 1.0,
            "profile_overview": 1.15,
            ResumeSectionType.EDUCATION.value: 0.9,
            ResumeSectionType.CERTIFICATIONS.value: 1.0,
            ResumeSectionType.OTHER.value: 0.8
        }

    def clear(self):
        """Clears all indexed chunks and embeddings."""
        self.chunks = []
        self.dense_embeddings = None
        self.corpus_tokens = []

    def add_chunks(self, chunks: List[ChunkDocument]):
        """Indexes new document chunks and generates dense embeddings + BM25 tokens."""
        if not chunks:
            return

        texts = [chunk.content for chunk in chunks]
        new_embeddings = self.embedding_manager.embed_batch(texts)
        new_dense_array = np.array(new_embeddings, dtype=np.float32)

        if self.dense_embeddings is None or len(self.chunks) == 0:
            self.dense_embeddings = new_dense_array
        else:
            self.dense_embeddings = np.vstack([self.dense_embeddings, new_dense_array])

        for chunk in chunks:
            self.chunks.append(chunk)
            self.corpus_tokens.append(self._tokenize(chunk.content))

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.65,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[ChunkDocument]:
        """
        Executes hybrid search combining dense semantic similarity and BM25 sparse keyword scores.
        
        Args:
            query: Free-text search query or Job Description snippet
            top_k: Number of top results to return
            alpha: Weight for dense vs sparse (1.0 = Pure Dense, 0.0 = Pure BM25)
            filters: Optional metadata filters (e.g. {'candidate_id': 'xyz', 'min_years': 4})
        """
        if not self.chunks or self.dense_embeddings is None:
            return []

        # 1. Apply Metadata Filters
        valid_indices = self._filter_indices(filters)
        if not valid_indices:
            return []

        # 2. Dense Semantic Search (Cosine Similarity)
        query_emb = np.array(self.embedding_manager.embed_text(query), dtype=np.float32)
        q_norm = np.linalg.norm(query_emb)
        if q_norm > 0:
            query_emb = query_emb / q_norm

        doc_norms = np.linalg.norm(self.dense_embeddings[valid_indices], axis=1, keepdims=True)
        doc_norms[doc_norms == 0] = 1e-10
        norm_doc_embeddings = self.dense_embeddings[valid_indices] / doc_norms
        dense_scores = np.dot(norm_doc_embeddings, query_emb)
        # Normalize dense scores to [0, 1]
        dense_min, dense_max = dense_scores.min(), dense_scores.max()
        if dense_max > dense_min:
            dense_norm_scores = (dense_scores - dense_min) / (dense_max - dense_min)
        else:
            dense_norm_scores = np.ones_like(dense_scores) * 0.5

        # 3. Sparse BM25 Keyword Search
        query_tokens = self._tokenize(query)
        bm25_scores = self._compute_bm25(query_tokens, valid_indices)
        bm25_min, bm25_max = bm25_scores.min(), bm25_scores.max()
        if bm25_max > bm25_min:
            bm25_norm_scores = (bm25_scores - bm25_min) / (bm25_max - bm25_min)
        else:
            bm25_norm_scores = np.zeros_like(bm25_scores)

        # 4. Combine Scores with Section Weighting
        results: List[Tuple[float, ChunkDocument]] = []
        for i, original_idx in enumerate(valid_indices):
            chunk = self.chunks[original_idx]
            sec_type = chunk.metadata.get("section_type", chunk.section_type.value if hasattr(chunk.section_type, "value") else str(chunk.section_type))
            sec_weight = self.section_weights.get(sec_type, 1.0)

            d_score = float(dense_norm_scores[i])
            s_score = float(bm25_norm_scores[i])
            hybrid_score = (alpha * d_score + (1.0 - alpha) * s_score) * sec_weight

            chunk_copy = chunk.model_copy()
            chunk_copy.dense_score = d_score
            chunk_copy.sparse_score = s_score
            chunk_copy.combined_score = round(hybrid_score, 4)
            results.append((hybrid_score, chunk_copy))

        # Sort descending by combined score
        results.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in results[:top_k]]

    def search_for_candidate(self, candidate_id: str, query: str, top_k: int = 4) -> List[ChunkDocument]:
        """Retrieves most relevant chunks specifically for a single candidate."""
        return self.hybrid_search(query=query, top_k=top_k, filters={"candidate_id": candidate_id})

    def _filter_indices(self, filters: Optional[Dict[str, Any]]) -> List[int]:
        """Evaluates metadata filters and returns matching indices."""
        if not filters:
            return list(range(len(self.chunks)))

        indices = []
        for idx, chunk in enumerate(self.chunks):
            match = True
            meta = chunk.metadata

            if "candidate_id" in filters and meta.get("candidate_id") != filters["candidate_id"]:
                match = False
            if "section_type" in filters and meta.get("section_type") != filters["section_type"]:
                match = False
            if "min_years" in filters:
                years = meta.get("years_experience", 0.0) or 0.0
                if float(years) < float(filters["min_years"]):
                    match = False

            if match:
                indices.append(idx)
        return indices

    def _tokenize(self, text: str) -> List[str]:
        """Extracts normalized lowercase alphanumeric tokens."""
        return re.findall(r"\b[a-zA-Z0-9_+#.-]+\b", text.lower())

    def _compute_bm25(self, query_tokens: List[str], valid_indices: List[int]) -> np.ndarray:
        """Computes BM25 Okapi scores for the query against indexed documents."""
        k1 = 1.5
        b = 0.75
        N = len(valid_indices)
        if N == 0 or not query_tokens:
            return np.zeros(N, dtype=np.float32)

        # Average document length for filtered corpus
        doc_lens = [len(self.corpus_tokens[idx]) for idx in valid_indices]
        avg_dl = sum(doc_lens) / max(1, N)

        # Calculate Document Frequencies (DF) for query tokens
        scores = np.zeros(N, dtype=np.float32)
        for token in set(query_tokens):
            df = sum(1 for idx in valid_indices if token in self.corpus_tokens[idx])
            if df == 0:
                continue

            # IDF calculation with smoothing
            idf = math.log(1.0 + (N - df + 0.5) / (df + 0.5))

            for i, idx in enumerate(valid_indices):
                tf = self.corpus_tokens[idx].count(token)
                if tf > 0:
                    doc_len = doc_lens[i]
                    term_score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_dl)))
                    scores[i] += term_score

        return scores
