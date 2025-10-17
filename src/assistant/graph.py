"""
Autor: Luis González Fraga

Asunciones:
- Requiere Python >= 3.13
"""
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from langchain_core.runnables.config import RunnableConfig  # typing correcto

from src.assistant.utils import (
    invoke_ollama,
    web_search,                   # <-- router unificado
    format_documents_with_metadata,
)
from src.assistant.vector_db import get_or_create_vector_db


# -----------------------------
# Estado del grafo
# -----------------------------
class GraphState(TypedDict, total=False):
    user_instructions: str
    queries: List[str]
    web_summaries: List[str]   # para la GUI
    web_context: str           # corpus web [W#] con título, url, snippet
    retrieved_docs: List[Document]
    final_answer: str


# -----------------------------
# Prompts base
# -----------------------------
QUERY_WRITER_SYSTEM = (
    "Eres un asistente que crea 3-5 consultas de investigación claras, "
    "breves y distintas entre sí, en español. No incluyas explicaciones, "
    "solo una lista JSON bajo la clave 'queries'."
)

FINAL_ANSWER_SYSTEM_RAG_ONLY = (
    "Eres un asistente que RESPONDE ÚNICAMENTE con la información del CONTEXTO proporcionado.\n"
    "- Si la respuesta no está explícitamente en el CONTEXTO, responde exactamente: "
    "\"No hay suficiente información en los documentos.\"\n"
    "- No inventes datos. No uses conocimiento externo. No hagas suposiciones.\n"
    "- Si tu modelo genera un bloque de razonamiento, consérvalo entre <think>...</think> (breve) antes de la respuesta.\n"
    "- Usa citas [S#] en cada afirmación relevante."
)

FINAL_ANSWER_SYSTEM_WEB_ONLY = (
    "Eres un asistente que RESPONDE ÚNICAMENTE con la información de los RESÚMENES_WEB proporcionados.\n"
    "- Si la respuesta no está explícitamente en los RESÚMENES_WEB, responde exactamente: "
    "\"No hay suficiente información en las fuentes web recuperadas.\"\n"
    "- No inventes datos. No uses conocimiento externo. No hagas suposiciones.\n"
    "- Si tu modelo genera un bloque de razonamiento, consérvalo entre <think>...</think> (breve) antes de la respuesta.\n"
    "- Usa citas [W#] en cada afirmación relevante."
)

FINAL_ANSWER_SYSTEM_RAG_PLUS_WEB = (
    "Eres un asistente que redacta una respuesta en español priorizando el CONTEXTO interno [S#] "
    "y complementando con RESÚMENES_WEB [W#] cuando sea útil.\n"
    "- No inventes datos. Si algo no está en [S#] o [W#], dilo claramente.\n"
    "- Si tu modelo genera un bloque de razonamiento, consérvalo entre <think>...</think> (breve) antes de la respuesta.\n"
    "- Cita [S#] y [W#] según corresponda."
)


# -----------------------------
# Nodos del grafo
# -----------------------------
def generate_research_queries(state: GraphState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Genera consultas a partir de la instrucción del usuario."""
    user_ins = state["user_instructions"]
    cfg = config.get("configurable", {}) if config else {}
    model = cfg.get("ollama_model", "qwen3:14b")
    max_q = int(cfg.get("max_search_queries", 5))

    user_prompt = (
        f"Usuario:\n{user_ins}\n\n"
        f"Genera entre 3 y {max_q} consultas diversas y útiles (JSON con 'queries')."
    )

    try:
        from pydantic import BaseModel

        class Queries(BaseModel):
            queries: List[str]

        result = invoke_ollama(
            model=model,
            system_prompt=QUERY_WRITER_SYSTEM,
            user_prompt=user_prompt,
            output_format=Queries
        )
        queries = [q.strip() for q in result.queries if q.strip()]
    except Exception:
        # Fallback: con un parseo indulgente
        raw = invoke_ollama(model, QUERY_WRITER_SYSTEM, user_prompt)
        import json
        try:
            obj = json.loads(raw)
            queries = [q.strip() for q in obj.get("queries", []) if isinstance(q, str) and q.strip()]
        except Exception:
            candidates = [s.strip("-• ").strip() for s in raw.splitlines() if s.strip()]
            queries = [c for c in candidates if len(c) > 3][:max_q]
        if not queries:
            queries = [user_ins]

    pretty = "### Consultas de investigación\n" + "\n".join(f"- {q}" for q in queries)
    return {"queries": queries, "generate_research_queries": pretty}


def search_and_summarize_query(state: GraphState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """
    Búsqueda web según el modo:
      - source_mode = 'rag'   -> deshabilitada
      - source_mode = 'web'   -> habilitada
      - source_mode = 'hybrid'-> habilitada
    Devuelve:
      - web_summaries: líneas cortas para GUI
      - web_context:   texto con snippets (para el prompt final)
    """
    cfg = config.get("configurable", {}) if config else {}
    source_mode = cfg.get("source_mode", "hybrid")
    provider = cfg.get("search_provider", "tavily")
    enable_web = (source_mode != "rag")
    queries = state.get("queries", [])

    if not queries:
        return {
            "web_summaries": [],
            "web_context": "",
            "search_and_summarize_query": "No hay consultas para buscar."
        }

    if not enable_web:
        return {
            "web_summaries": [],
            "web_context": "",
            "search_and_summarize_query": "Búsqueda web deshabilitada (modo Solo RAG)."
        }

    summaries: List[str] = []
    blocks_for_gui: List[str] = []
    blocks_for_prompt: List[str] = []

    for i, q in enumerate(queries, start=1):
        try:
            res = web_search(q, provider=provider, max_results=int(cfg.get("max_search_queries", 5)))
            items = res.get("results", []) if isinstance(res, dict) else []

            gui_lines = [f"**[W{i}] Query:** {q} (via {provider})"]
            prompt_lines = [f"[W{i}] Query: {q} (provider: {provider})"]

            if not items:
                gui_lines.append("Sin resultados.")
                summaries.append(f"[W{i}] {q}: sin resultados.")
            else:
                summaries.append(f"[W{i}] {q}: {len(items)} resultados.")
                for j, it in enumerate(items, start=1):
                    title = it.get("title") or "(sin título)"
                    url = it.get("url") or ""
                    content = (it.get("content") or "").strip()
                    snippet = content[:800]
                    gui_lines.append(f"[{j}] {title}\n{url}\n{snippet[:400]}...")
                    prompt_lines.append(
                        f"- ({j}) {title}\n  URL: {url}\n  Extracto: {snippet}"
                    )

            blocks_for_gui.append("\n\n".join(gui_lines))
            blocks_for_prompt.append("\n".join(prompt_lines))

        except Exception as e:
            msg = f"Error en búsqueda ({provider}): {e}"
            blocks_for_gui.append(f"**[W{i}] Query:** {q}\n{msg}")
            summaries.append(f"[W{i}] {q}: error en la búsqueda.")

    pretty_gui = "### Resúmenes de búsqueda web\n\n" + "\n\n---\n\n".join(blocks_for_gui)
    web_context_text = "\n\n".join(blocks_for_prompt)

    return {
        "web_summaries": summaries,
        "web_context": web_context_text,
        "search_and_summarize_query": pretty_gui
    }


def retrieve_rag_documents(state: GraphState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """
    Recupera documentos RAG según el modo:
      - source_mode = 'web'  -> deshabilitada
      - 'rag' / 'hybrid'     -> habilitada
    """
    cfg = config.get("configurable", {}) if config else {}
    source_mode = cfg.get("source_mode", "hybrid")

    if source_mode == "web":
        return {
            "retrieved_docs": [],
            "retrieve_rag_documents": "Recuperación RAG deshabilitada (modo Solo Web)."
        }

    queries = state.get("queries", []) or [state.get("user_instructions", "")]
    vectorstore = get_or_create_vector_db()

    try:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6}
        )
    except Exception:
        retriever = vectorstore.as_retriever()

    collected: List[Document] = []
    for q in queries:
        try:
            docs = retriever.invoke(q)
        except Exception:
            docs = retriever.get_relevant_documents(q)

        for d in docs:
            sig = (d.metadata.get("source"), d.page_content[:80])
            if not any((x.metadata.get("source"), x.page_content[:80]) == sig for x in collected):
                collected.append(d)

    sample = collected[:5]
    pretty_block = "### Recuperación de documentos (muestra)\n\n" + format_documents_with_metadata(sample)
    return {"retrieved_docs": collected, "retrieve_rag_documents": pretty_block}


def generate_final_answer(state: GraphState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Redacta la respuesta final respetando el modo seleccionado y LA PLANTILLA."""
    cfg = config.get("configurable", {}) if config else {}
    model = cfg.get("ollama_model", "qwen3:14b")
    source_mode = cfg.get("source_mode", "hybrid")
    report_structure = cfg.get("report_structure", "")
    answer_length = cfg.get("answer_length", "medium")   # 'short'|'medium'|'long'

    user_q = state.get("user_instructions", "")
    docs = state.get("retrieved_docs", []) or []

    # CONTEXTO RAG (S#)
    if docs:
        blocks = []
        for i, d in enumerate(docs[:12], start=1):
            src = d.metadata.get("filename") or d.metadata.get("source") or "desconocido"
            txt = (d.page_content or "")[:1800]
            blocks.append(f"[S{i}] Fuente: {src}\n{txt}")
        contexto = "\n\n".join(blocks)
    else:
        contexto = ""

    # CONTEXTO WEB (W#)
    web_context = state.get("web_context", "") or ""

    # Reglas de longitud
    length_map = {
        "short":  "Extensión breve (1–2 párrafos, 6–10 líneas en total).",
        "medium": "Extensión media (3–5 párrafos).",
        "long":   "Extensión amplia (desarrolla secciones completas con detalles).",
    }
    length_instr = length_map.get(answer_length, length_map["medium"])

    # PLANTILLA: reforzada y obligatoria
    plantilla = (
        "Sigue y respeta EXACTAMENTE la siguiente PLANTILLA de informe. "
        "Completa cada encabezado con contenido. Si una sección no puede completarse con las fuentes permitidas, "
        "indícalo explícitamente en esa sección.\n\n"
        f"{report_structure}\n\n"
        f"Requisitos de extensión: {length_instr}\n"
        "Cita con [S#] para CONTEXTO y [W#] para Web según proceda."
    )

    # Prompts por modo
    if source_mode == "rag":
        system_prompt = FINAL_ANSWER_SYSTEM_RAG_ONLY
        user_prompt = (
            f"Pregunta del usuario:\n{user_q}\n\n"
            f"CONTEXTO (fragmentos [S#]):\n{contexto}\n\n"
            f"{plantilla}"
        )
    elif source_mode == "web":
        system_prompt = FINAL_ANSWER_SYSTEM_WEB_ONLY
        user_prompt = (
            f"Pregunta del usuario:\n{user_q}\n\n"
            f"RESÚMENES_WEB (corpus [W#]):\n{web_context}\n\n"
            f"{plantilla}"
        )
    else:  # hybrid
        system_prompt = FINAL_ANSWER_SYSTEM_RAG_PLUS_WEB
        user_prompt = (
            f"Pregunta del usuario:\n{user_q}\n\n"
            f"CONTEXTO (fragmentos [S#]):\n{contexto}\n\n"
            f"RESÚMENES_WEB (corpus [W#]):\n{web_context}\n\n"
            f"{plantilla}"
        )

    try:
        answer = invoke_ollama(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_format=None
        )
    except Exception as e:
        answer = f"(Error al generar la respuesta con el modelo '{model}'): {e}"

    return {"final_answer": answer, "generate_final_answer": answer}


# -----------------------------
# Compilación del grafo
# -----------------------------
builder = StateGraph(GraphState)
builder.add_node("generate_research_queries", generate_research_queries)
builder.add_node("search_and_summarize_query", search_and_summarize_query)
builder.add_node("retrieve_rag_documents", retrieve_rag_documents)
builder.add_node("generate_final_answer", generate_final_answer)

builder.set_entry_point("generate_research_queries")
builder.add_edge("generate_research_queries", "search_and_summarize_query")
builder.add_edge("search_and_summarize_query", "retrieve_rag_documents")
builder.add_edge("retrieve_rag_documents", "generate_final_answer")
builder.add_edge("generate_final_answer", END)

researcher = builder.compile()
