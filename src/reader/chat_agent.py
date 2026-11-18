"""Conversational RAG agent with source grounding and citation generation."""
from typing import List, Optional, Dict, Any
from src.models.schemas import (
    ChatResponse,
    Citation,
    CandidateProfile,
    ChunkDocument,
    ResumeSectionType
)
from src.evaluator.llm_client import LLMClient
from src.retrieval.vector_store import HybridVectorStore
from src.retrieval.reranker import Reranker


class ResumeChatAgent:
    """Conversational intelligence agent for querying the candidate pool or deep-diving into individual resumes."""

    def __init__(
        self,
        vector_store: HybridVectorStore,
        candidates: Optional[Dict[str, CandidateProfile]] = None,
        llm_client: Optional[LLMClient] = None
    ):
        self.vector_store = vector_store
        self.candidates = candidates or {}
        self.llm = llm_client or LLMClient()

    def update_candidates(self, candidates: Dict[str, CandidateProfile]):
        """Refreshes the internal candidate profile map."""
        self.candidates = candidates

    def chat(
        self,
        query: str,
        focused_candidate_id: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = 5
    ) -> ChatResponse:
        """
        Processes a user question, retrieves relevant resume context, and returns a grounded answer with citations.
        """
        # 1. Retrieve Candidate Chunks
        filters = {"candidate_id": focused_candidate_id} if focused_candidate_id else None
        retrieved_chunks = self.vector_store.hybrid_search(
            query=query,
            top_k=top_k * 2,
            alpha=0.65,
            filters=filters
        )

        # 2. Re-rank retrieved chunks for maximum precision
        reranked_chunks = Reranker.rerank(query=query, chunks=retrieved_chunks, top_k=top_k)

        # 3. Build Citations list
        citations: List[Citation] = []
        referenced_candidate_ids = set()

        for chunk in reranked_chunks:
            c_id = chunk.candidate_id
            c_name = chunk.candidate_name
            referenced_candidate_ids.add(c_id)

            snippet = chunk.content.replace(f"Candidate: {c_name}\n", "").replace(f"Section: {chunk.section_type.value.upper()}\n\n", "")
            snippet = snippet[:220].strip() + ("..." if len(snippet) > 220 else "")

            citations.append(
                Citation(
                    candidate_id=c_id,
                    candidate_name=c_name,
                    section_type=chunk.section_type.value if hasattr(chunk.section_type, "value") else str(chunk.section_type),
                    snippet=snippet,
                    relevance_score=chunk.combined_score
                )
            )

        # 4. Generate Answer via LLM
        if self.llm.is_live_llm_available and reranked_chunks:
            answer = self._generate_llm_answer(query, reranked_chunks, chat_history, focused_candidate_id)
        else:
            answer = self._generate_extractive_answer(query, reranked_chunks, focused_candidate_id)

        return ChatResponse(
            answer=answer,
            citations=citations,
            candidate_ids_referenced=list(referenced_candidate_ids)
        )

    def _generate_llm_answer(
        self,
        query: str,
        chunks: List[ChunkDocument],
        chat_history: Optional[List[Dict[str, str]]],
        focused_candidate_id: Optional[str]
    ) -> str:
        """Generates grounded answer using LLM."""
        system_prompt = (
            "You are an AI Recruitment Intelligence Assistant. "
            "Your task is to answer recruiter queries accurately based STRICTLY on the provided resume context snippets. "
            "Cite candidate names and specific resume sections where appropriate. "
            "If the information is not present in the resumes, state clearly that it is not found in the documents."
        )

        context_blocks = []
        for i, chunk in enumerate(chunks):
            context_blocks.append(f"--- [Document #{i+1} | {chunk.candidate_name} | {chunk.section_type.value.upper()}] ---\n{chunk.content}")
        context_str = "\n\n".join(context_blocks)

        history_str = ""
        if chat_history:
            history_str = "\nRecent Conversation:\n" + "\n".join(f"{h['role'].title()}: {h['content']}" for h in chat_history[-3:])

        prompt = (
            f"Resume Context Snippets:\n{context_str}\n"
            f"{history_str}\n\n"
            f"User Question: {query}\n\n"
            f"Provide a concise, direct, and well-structured answer based only on the evidence above:"
        )

        resp = self.llm.generate_text(prompt=prompt, system_prompt=system_prompt)
        return resp if resp else self._generate_extractive_answer(query, chunks, focused_candidate_id)

    def _generate_extractive_answer(
        self,
        query: str,
        chunks: List[ChunkDocument],
        focused_candidate_id: Optional[str]
    ) -> str:
        """Intelligent extractive synthesis when live LLM API is not reachable."""
        if not chunks:
            if focused_candidate_id and focused_candidate_id in self.candidates:
                cand = self.candidates[focused_candidate_id]
                return f"No direct matches found in {cand.name}'s resume for '{query}'. Their listed skills include: {', '.join(cand.top_skills[:8])}."
            return f"No candidate matches found in the current resume pool for the query: '{query}'."

        answer_lines = [f"Found **{len(chunks)} relevant resume sections** matching your query:"]
        for idx, chunk in enumerate(chunks, 1):
            cand_name = chunk.candidate_name
            sec_name = chunk.metadata.get("section_type", "Section").replace("_", " ").title()
            clean_snippet = chunk.content.split("\n\n", 1)[-1].strip()
            first_few_lines = "\n".join(clean_snippet.splitlines()[:3])
            
            score_str = f"(Match Confidence: {int((chunk.combined_score or 0.7) * 100)}%)"
            answer_lines.append(f"\n**{idx}. {cand_name}** - *{sec_name}* {score_str}\n> {first_few_lines}")

        return "\n".join(answer_lines)
