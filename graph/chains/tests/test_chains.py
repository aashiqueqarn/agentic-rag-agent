from dotenv import load_dotenv
from pprint import pprint

from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.generation import generation_chain
from ingestion import retriver

load_dotenv()

def test_retrieval_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriver.invoke(question)
    doc_text = docs[0].page_content
    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_text}
    )
    assert res.binary_score == "yes"


def test_retrieval_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriver.invoke(question)
    doc_text = docs[0].page_content
    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "How to make pizza", "document": doc_text}
    )
    assert res.binary_score == "no"

def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriver.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)