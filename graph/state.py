from typing import List, TypedDict

class GraphState(TypedDict):
    """
    Represents the state of our graph

    Attributes:
        question: question
        generation: generation
        web_search: weather to add search
        documents: list of documents
    """
    question: str
    generation: str
    web_search: bool
    documents: List[str]