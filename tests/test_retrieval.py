"""Unit tests for Embeddings, Hybrid Vector Store, and Re-ranking."""
from src.ingestion.sectionizer import ResumeSectionizer
from src.retrieval.embeddings import EmbeddingManager
from src.retrieval.vector_store import HybridVectorStore
from src.retrieval.reranker import Reranker

RESUME_AI = """
Maya Lin
Seattle, WA | maya@example.com
PROFESSIONAL SUMMARY
Lead AI/ML and RAG Systems Engineer with 6+ years experience in PyTorch, LangChain, Vector Databases, and FastAPI.
WORK EXPERIENCE
AI Engineer | TechCorp
2020 - Present
Built enterprise RAG search platform using ChromaDB and LangChain.
SKILLS
Python, PyTorch, LangChain, ChromaDB, FastAPI, Docker
"""

RESUME_DEVOPS = """
Sarah Connor
Austin, TX | sarah@example.com
PROFESSIONAL SUMMARY
DevOps Engineer with 8+ years experience in AWS, Kubernetes, Docker, Terraform, and CI/CD.
WORK EXPERIENCE
DevOps Lead | CloudNet
2018 - Present
Deployed AWS EKS Kubernetes clusters and Terraform infrastructure.
SKILLS
AWS, Kubernetes, Terraform, Docker, CI/CD, Linux, Bash
"""


def test_embedding_generation():
    embedder = EmbeddingManager(provider="mock")
    vec = embedder.embed_text("Machine Learning and Vector Databases")
    assert isinstance(vec, list)
    assert len(vec) == 256
    
    # Non-empty and normalized
    assert any(v != 0.0 for v in vec)


def test_hybrid_search_and_ranking():
    store = HybridVectorStore()
    
    # Ingest 2 candidate profiles
    p1 = ResumeSectionizer.sectionize(RESUME_AI, filename="maya.txt")
    p2 = ResumeSectionizer.sectionize(RESUME_DEVOPS, filename="sarah.txt")
    
    store.add_chunks(ResumeSectionizer.create_chunks(p1))
    store.add_chunks(ResumeSectionizer.create_chunks(p2))
    
    # 1. Search for AI / RAG query
    ai_results = store.hybrid_search("Vector databases and RAG LangChain", top_k=3)
    assert len(ai_results) > 0
    assert ai_results[0].candidate_name == "Maya Lin"

    # 2. Search for DevOps / Kubernetes query
    devops_results = store.hybrid_search("Kubernetes EKS Terraform AWS cloud", top_k=3)
    assert len(devops_results) > 0
    assert devops_results[0].candidate_name == "Sarah Connor"


def test_metadata_filtering():
    store = HybridVectorStore()
    p1 = ResumeSectionizer.sectionize(RESUME_AI, filename="maya.txt")
    p2 = ResumeSectionizer.sectionize(RESUME_DEVOPS, filename="sarah.txt")
    store.add_chunks(ResumeSectionizer.create_chunks(p1))
    store.add_chunks(ResumeSectionizer.create_chunks(p2))

    # Filter strictly for Maya Lin
    maya_only = store.hybrid_search("Cloud and Docker", top_k=5, filters={"candidate_id": p1.candidate_id})
    assert all(c.candidate_id == p1.candidate_id for c in maya_only)


def test_reranker():
    store = HybridVectorStore()
    p1 = ResumeSectionizer.sectionize(RESUME_AI, filename="maya.txt")
    chunks = ResumeSectionizer.create_chunks(p1)
    store.add_chunks(chunks)

    retrieved = store.hybrid_search("PyTorch and LangChain", top_k=4)
    reranked = Reranker.rerank("PyTorch and LangChain", retrieved, top_k=2)
    
    assert len(reranked) <= 2
    assert reranked[0].combined_score is not None
