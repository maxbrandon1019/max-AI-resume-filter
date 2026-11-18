"""Unit tests for Resume Conversational Reader Agent."""
from src.ingestion.sectionizer import ResumeSectionizer
from src.retrieval.vector_store import HybridVectorStore
from src.reader.chat_agent import ResumeChatAgent

RESUME_1 = """
Alex Rivera
alex@example.com
PROFESSIONAL SUMMARY
Senior Backend Engineer with 8 years experience in Go, Python, and Kafka.
WORK EXPERIENCE
Senior Engineer | CloudScale
2021 - Present
Architected event-driven microservices using Python FastAPI, Go, and Apache Kafka.
SKILLS
Go, Python, FastAPI, Kafka, Redis, PostgreSQL
"""


def test_chat_agent_citations():
    store = HybridVectorStore()
    profile = ResumeSectionizer.sectionize(RESUME_1, filename="alex.txt")
    store.add_chunks(ResumeSectionizer.create_chunks(profile))

    agent = ResumeChatAgent(vector_store=store, candidates={profile.candidate_id: profile})
    
    response = agent.chat(query="What message broker does Alex use?")
    
    assert response is not None
    assert len(response.citations) > 0
    assert response.citations[0].candidate_name == "Alex Rivera"
    assert profile.candidate_id in response.candidate_ids_referenced


def test_chat_agent_scoped_candidate():
    store = HybridVectorStore()
    profile = ResumeSectionizer.sectionize(RESUME_1, filename="alex.txt")
    store.add_chunks(ResumeSectionizer.create_chunks(profile))

    agent = ResumeChatAgent(vector_store=store, candidates={profile.candidate_id: profile})
    response = agent.chat(query="Summarize experience", focused_candidate_id=profile.candidate_id)

    assert response is not None
    assert len(response.citations) > 0
    assert response.citations[0].candidate_id == profile.candidate_id
