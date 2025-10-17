"""
Autor: Luis González Fraga
Utilidades para la GUI
Asunciones:
- Requiere Python >= 3.13
"""
import os
import re
import shutil
import warnings
from typing import List, Dict, Any

import requests
from ollama import Client, chat as ollama_chat
from tavily import TavilyClient
from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_community.document_loaders import CSVLoader, PDFPlumberLoader
from src.assistant.vector_db import add_documents


# =========================
# Modelos Pydantic auxiliares
# =========================
class Evaluation(BaseModel):
    is_relevant: bool


class Queries(BaseModel):
    queries: list[str]


# =========================
# Utilidades generales
# =========================
def parse_output(text: str):
    think = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    output_match = re.search(r'</think>\s*(.*?)$', text, re.DOTALL)
    reasoning = think.group(1).strip() if think else None
    output = output_match.group(1).strip() if output_match else text.strip()
    return {"reasoning": reasoning, "response": output}


def format_documents_with_metadata(documents: List[Document]) -> str:
    formatted_docs = []
    for doc in documents:
        source = doc.metadata.get("source", "Unknown source")
        formatted_doc = f"Source: {source}\nContent: {doc.page_content}"
        formatted_docs.append(formatted_doc)
    return "\n\n---\n\n".join(formatted_docs)


def _ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST") or "http://127.0.0.1:11434"


# ---- Reutiliza un único cliente de Ollama para evitar sockets abiertos
_OLLAMA_CLIENT: Client | None = None
def _get_ollama_client() -> Client:
    global _OLLAMA_CLIENT
    if _OLLAMA_CLIENT is None:
        _OLLAMA_CLIENT = Client(host=_ollama_base_url())
    return _OLLAMA_CLIENT


def invoke_ollama(model, system_prompt, user_prompt, output_format=None):
    client = _get_ollama_client()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        resp = client.chat(
            model=model,
            messages=messages,
            format=output_format.model_json_schema() if output_format else None,
         
        )
    except Exception as e:
        raise RuntimeError(
            f"No se pudo conectar con Ollama en '{_ollama_base_url()}'. "
            f"Verifica que el servicio está activo. Detalle: {e}"
        ) from e

    # Unificar acceso (dict u objeto)
    def _get(obj, path, default=None):
        cur = obj
        for key in path:
            if isinstance(cur, dict):
                cur = cur.get(key, default)
            else:
                cur = getattr(cur, key, default)
            if cur is default:
                break
        return cur

    content = _get(resp, ["message", "content"], "") or ""

    if output_format:
        # structured output
        return output_format.model_validate_json(content)

    possible = []

    # message.thinking / message.reasoning (string)
    for k in ("thinking", "reasoning", "thought", "chain_of_thought"):
        val = _get(resp, ["message", k], None)
        if isinstance(val, str) and val.strip():
            possible.append(val.strip())

    # meta.reasoning.output_text / meta.reasoning.content
    meta_reason = _get(resp, ["meta", "reasoning"], None)
    if isinstance(meta_reason, dict):
        for k in ("output_text", "content"):
            val = meta_reason.get(k)
            if isinstance(val, str) and val.strip():
                possible.append(val.strip())

    # top-level reasoning como string (por si acaso)
    top_reason = _get(resp, ["reasoning"], None)
    if isinstance(top_reason, str) and top_reason.strip():
        possible.append(top_reason.strip())

    # Si el content aún no tiene <think>…, lo añadimos
    if not re.search(r"<think>.*?</think>", content, flags=re.DOTALL | re.IGNORECASE):
        joined = "\n".join(p.strip() for p in possible if p.strip())
        if joined:
            content = f"<think>{joined}</think>\n{content}"

    return content


def invoke_llm(model, system_prompt, user_prompt, output_format=None, temperature=0):
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
    )
    if output_format:
        llm = llm.with_structured_output(output_format)

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    response = llm.invoke(messages)
    if output_format:
        return response
    return response.content


# =========================
# PROVEEDORES DE BÚSQUEDA WEB
# =========================
def tavily_search(query: str, include_raw_content: bool = True, max_results: int = 3) -> Dict[str, Any]:
    client = TavilyClient()
    return client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content
    )


def _unify(items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, str]]]:
    """Normaliza items {'title','url','content'} -> {'results':[...]}"""
    norm = []
    for it in items:
        title = it.get("title") or it.get("t") or "(sin título)"
        url = it.get("url") or it.get("link") or it.get("href") or ""
        content = it.get("content") or it.get("snippet") or it.get("body") or ""
        norm.append({"title": str(title), "url": str(url), "content": str(content)})
    return {"results": norm}


def _search_tavily(query: str, max_results: int = 5) -> Dict[str, Any]:
    res = tavily_search(query, include_raw_content=False, max_results=max_results)
    items = []
    for r in res.get("results", []):
        items.append({
            "title": r.get("title"),
            "url": r.get("url"),
            "content": (r.get("content") or "")[:1200],
        })
    return _unify(items)


def _search_google_cse(query: str, max_results: int = 5) -> Dict[str, Any]:
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    cse_id  = os.getenv("GOOGLE_CSE_ID", "").strip()
    if not api_key or not cse_id:
        raise RuntimeError("Faltan GOOGLE_API_KEY o GOOGLE_CSE_ID en el entorno (.env).")

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key, "cx": cse_id, "q": query,
        "num": max(1, min(max_results, 10)),
        "safe": "off", "hl": "es",
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    items = []
    for it in data.get("items", []):
        items.append({
            "title": it.get("title"),
            "url": it.get("link"),
            "content": it.get("snippet", "")[:1200],
        })
    return _unify(items)


def _search_serper(query: str, max_results: int = 5) -> Dict[str, Any]:
    api_key = os.getenv("SERPER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Falta SERPER_API_KEY en el entorno (.env).")

    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {"q": query, "num": max(1, min(max_results, 10))}
    r = requests.post(url, headers=headers, json=payload, timeout=20)
    r.raise_for_status()
    data = r.json()

    items = []
    for it in data.get("organic", []):
        items.append({
            "title": it.get("title"),
            "url": it.get("link"),
            "content": it.get("snippet", "")[:1200],
        })
    return _unify(items)


def _search_ddg(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    DuckDuckGo con el paquete nuevo `ddgs`.
    Fallback transparente a `duckduckgo_search` si aún lo tienes instalado.
    """
    try:
        from ddgs import DDGS  # paquete nuevo
    except Exception:
        try:
            # Fallback (muestra deprec. del paquete antiguo). Silenciamos ese warning.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="This package \\(`duckduckgo_search`\\) has been renamed to `ddgs`!",
                    category=RuntimeWarning
                )
                from duckduckgo_search import DDGS  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Para usar DuckDuckGo instala el paquete 'ddgs' (pip install ddgs). "
                f"Detalle: {e}"
            ) from e

    items = []
    # Nota: API estable: ddgs.text(query, region, safesearch, timelimit, max_results)
    with DDGS() as ddgs:
        for r in ddgs.text(query, region="es-es", safesearch="moderate", timelimit=None, max_results=max_results):
            items.append({
                "title": r.get("title"),
                "url": r.get("href") or r.get("link"),
                "content": r.get("body", "")[:1200],
            })
    return _unify(items)


def web_search(query: str, provider: str = "tavily", max_results: int = 5) -> Dict[str, Any]:
    """
    Router unificado de búsqueda.
    Devuelve siempre {'results':[{'title','url','content'}, ...]}
    """
    provider = (provider or os.getenv("SEARCH_PROVIDER", "tavily")).lower()
    if provider in ("tavily", "tvly"):
        return _search_tavily(query, max_results=max_results)
    if provider in ("google", "gcs", "cse"):
        return _search_google_cse(query, max_results=max_results)
    if provider in ("serper", "serper.dev"):
        return _search_serper(query, max_results=max_results)
    if provider in ("duckduckgo", "ddg"):
        return _search_ddg(query, max_results=max_results)
    # Fallback por defecto
    return _search_tavily(query, max_results=max_results)


# =========================
# Carga de plantillas
# =========================
def get_report_structures(reports_folder="report_structures"):
    report_structures = {}
    os.makedirs(reports_folder, exist_ok=True)
    try:
        for filename in os.listdir(reports_folder):
            if filename.endswith((".md", ".txt")):
                report_name = os.path.splitext(filename)[0]
                file_path = os.path.join(reports_folder, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as file:
                        content = file.read()
                        report_structures[report_name] = {"content": content}
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")
    except Exception as e:
        print(f"Error accessing reports folder: {str(e)}")
    return report_structures


# =========================
# Ingesta de ficheros
# =========================
def _read_text_with_detection(file_path: str) -> str:
    import chardet

    with open(file_path, "rb") as f:
        raw = f.read()

    if raw.startswith(b"\xef\xbb\xbf"):
        try:
            text = raw.decode("utf-8-sig")
            return text.replace("\r\n", "\n").replace("\r", "\n")
        except UnicodeDecodeError:
            pass

    detection = chardet.detect(raw) or {}
    enc = (detection.get("encoding") or "").lower()

    if enc:
        try:
            text = raw.decode(enc)
            return text.replace("\r\n", "\n").replace("\r", "\n")
        except UnicodeDecodeError:
            pass

    try:
        text = raw.decode("utf-8")
        return text.replace("\r\n", "\n").replace("\r", "\n")
    except UnicodeDecodeError:
        pass

    for fallback in ("cp1252", "latin-1"):
        try:
            text = raw.decode(fallback)
            return text.replace("\r\n", "\n").replace("\r", "\n")
        except UnicodeDecodeError:
            continue

    text = raw.decode("utf-8", errors="ignore")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def process_uploaded_files(uploaded_files):
    temp_folder = "temp_files"
    os.makedirs(temp_folder, exist_ok=True)

    try:
        for uploaded_file in uploaded_files:
            file_extension = uploaded_file.name.split(".")[-1].lower()
            temp_file_path = os.path.join(temp_folder, uploaded_file.name)

            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            if file_extension in ["txt", "md"]:
                try:
                    text = _read_text_with_detection(temp_file_path)
                    doc = Document(
                        page_content=text,
                        metadata={"source": temp_file_path, "filename": uploaded_file.name, "type": file_extension},
                    )
                    add_documents([doc])
                except Exception as e:
                    raise RuntimeError(f"Error loading {temp_file_path}: {e}") from e

            elif file_extension == "csv":
                try:
                    loader = CSVLoader(temp_file_path, encoding="utf-8")
                    docs = loader.load()
                    for d in docs:
                        d.metadata.setdefault("source", temp_file_path)
                        d.metadata.setdefault("filename", uploaded_file.name)
                        d.metadata.setdefault("type", "csv")
                    add_documents(docs)
                except Exception as e:
                    raise RuntimeError(f"Error loading {temp_file_path}: {e}") from e

            elif file_extension == "pdf":
                try:
                    loader = PDFPlumberLoader(temp_file_path)
                    docs = loader.load()
                    for d in docs:
                        d.metadata.setdefault("source", temp_file_path)
                        d.metadata.setdefault("filename", uploaded_file.name)
                        d.metadata.setdefault("type", "pdf")
                    add_documents(docs)
                except Exception as e:
                    raise RuntimeError(f"Error loading {temp_file_path}: {e}") from e
            else:
                continue

        return True
    finally:
        shutil.rmtree(temp_folder, ignore_errors=True)
