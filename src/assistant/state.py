"""
Autor: Luis González Fraga

Asunciones:
- Requiere Python >= 3.13
"""
import operator
from typing_extensions import TypedDict
from typing import Annotated

# -------------------------------
# Estado global del "investigador"
# -------------------------------
class ResearcherState(TypedDict):
    user_instructions: str
    research_queries: list[str]
    search_summaries: Annotated[list, operator.add]
    current_position: int
    final_answer: str

# -------------------------------
# Interfaces de entrada/salida del estado global
# -------------------------------
class ResearcherStateInput(TypedDict):
    user_instructions: str


class ResearcherStateOutput(TypedDict):
    final_answer: str

# -------------------------------
# Estado de una búsqueda individual
# -------------------------------
class QuerySearchState(TypedDict):
    query: str
    web_search_results: list
    retrieved_documents: list
    are_documents_relevant: bool
    search_summaries: list[str]

# -------------------------------
# Interfaces de entrada/salida de la búsqueda individual
# -------------------------------
class QuerySearchStateInput(TypedDict):
    query: str

class QuerySearchStateOutput(TypedDict):
    query: str
    search_summaries: list[str]