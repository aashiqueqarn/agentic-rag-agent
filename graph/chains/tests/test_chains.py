from dotenv import load_dotenv
from pprint import pprint

from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.router import RouteQuery, question_router
from graph.chains.generation import generation_chain
from ingestion import retriver

from graph.chains.hallunication_grader import GradeHallucination, hallucination_grader

load_dotenv()

def test_retrieval_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriver.invoke(question)
    doc_text = docs[0].page_content
    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_text}
    )
    print(res.binary_score)
    assert res.binary_score in ["yes", "no"]


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


def test_hallucination_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriver.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    res: GradeHallucination = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )
    print(res.binary_score)
    assert res.binary_score == True

def test_hallucination_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriver.invoke(question)
    res: GradeHallucination = hallucination_grader.invoke(
        {
            "documents": docs,
            "generation": "In order to make pizza, you need to first start with the dough."
        }
    )
    print(res.binary_score)
    assert res.binary_score in [True, False]

def test_router_to_vectorstore() -> None:
    question = "agent memory"
    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "vectorstore"

def test_router_to_websearch() -> None:
    question = "how to make pizza"
    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "websearch"