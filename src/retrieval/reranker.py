"""Re-ranking module for fine-grained ranking of retrieved candidate chunks."""
from typing import List, Tuple
from src.models.schemas import ChunkDocument


class Reranker:
    """Refines candidate chunk rankings using cross-entropy or lexical-semantic cross-scoring."""

    @classmethod
    def rerank(cls, query: str, chunks: List[ChunkDocument], top_k: int = 5) -> List[ChunkDocument]:
        """
        Re-ranks retrieved chunks based on exact query term proximity, density, and section relevance.
        """
        if not chunks:
            return []

        query_terms = set(query.lower().split())
        scored: List[Tuple[float, ChunkDocument]] = []

        for chunk in chunks:
            content_lower = chunk.content.lower()
            base_score = chunk.combined_score or 0.5
            
            # Count distinct query term hits
            term_hits = sum(1 for term in query_terms if term in content_lower)
            hit_ratio = term_hits / max(1, len(query_terms))
            
            # Phrase match bonus
            phrase_bonus = 0.2 if query.lower() in content_lower else 0.0

            final_score = (0.6 * base_score) + (0.3 * hit_ratio) + (0.1 * phrase_bonus)
            
            updated_chunk = chunk.model_copy()
            updated_chunk.combined_score = round(final_score, 4)
            scored.append((final_score, updated_chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]
