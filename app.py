"""
Autor: Luis González Fraga

GUI (Interfaz Gráfica de Usuario) Ollama WebRAG Researcher
Asunciones:
- Requiere Python >= 3.13
"""

import os
import json
import urllib.request

import pyperclip
import streamlit as st
import streamlit_nested_layout
from dotenv import load_dotenv
from ollama import Client

from src.assistant.graph import researcher
from src.assistant.utils import get_report_structures, process_uploaded_files

load_dotenv()


# -----------------------------
# URL base de Ollama
# -----------------------------
def _ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST") or "http://127.0.0.1:11434"


# -----------------------------
# Listado de modelos Ollama
# -----------------------------
def get_installed_ollama_models():
    base_url = _ollama_base_url()

    # 1) Cliente oficial
    try:
        client = Client(host=base_url)
        data = client.list()
        items = data.get("models", []) if isinstance(data, dict) else getattr(data, "models", []) or []
        names = []
        for m in items:
            nm = m.get("name") if isinstance(m, dict) else (getattr(m, "name", None) or getattr(m, "model", None))
            if nm:
                names.append(nm)
        names = sorted(set(names), key=str.lower)
        if names:
            return names
    except Exception:
        pass

    # 2) Fallback HTTP
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=3) as resp:
            payload = json.loads(resp.read().decode("utf-8", "ignore"))
        items = payload.get("models", []) or []
        names = []
        for m in items:
            nm = m.get("name") or m.get("model")
            if nm:
                names.append(nm)
        names = sorted(set(names), key=str.lower)
        if names:
            return names
    except Exception:
        pass

    return []


# -----------------------------
# Tema al estilo de LLM Studio + estilos de salida
# -----------------------------
def inject_llmstudio_theme():
    st.markdown(
        """
        <style>
        :root {
            --bg: #0b1020;
            --bg2: #12172b;
            --text: #e5e7eb;
            --primary: #7c3aed;
            --primary-600: #6d28d9;
        }
        .stApp { background: var(--bg); color: var(--text); }
        header[data-testid="stHeader"] {
            background: linear-gradient(90deg, rgba(124,58,237,0.15), rgba(34,211,238,0.12));
        }
        [data-testid="stSidebar"] {
            background: var(--bg2);
            color: var(--text);
            border-right: 1px solid rgba(255,255,255,0.05);
        }
        .stMarkdown p, .st-emotion-cache-16idsys p { color: var(--text) !important; }
        .stTextInput input, .stNumberInput input, .stTextArea textarea {
            background: #0e1427 !important; color: var(--text) !important; border: 1px solid #1e2442 !important;
        }
        .stButton > button { background: var(--primary); color: #fff; border: 0; border-radius: 8px; }
        .stButton > button:hover { background: var(--primary-600); }
        .stStatus { background: #0e1427 !important; border: 1px solid #222b54 !important; }
        .stAlert  { background: #151a30 !important; border: 1px solid #263159 !important; color: var(--text) !important; }
        .stExpander { background: #0e1427 !important; border: 1px solid #1d2548 !important; }
        .stExpander summary { color: #c7d2fe !important; }
        .stChatMessage { background: #0e1427 !important; border: 1px solid #1f2750 !important; }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #c7d2fe; }

        /* Bloques de salida */
        .assistant-answer, .assistant-answer * { color: #ffffff !important; }
        .assistant-reasoning {
            color: #e5e7eb !important;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
            white-space: pre-wrap;
            line-height: 1.4;
        }
        .assistant-reasoning code, .assistant-reasoning pre { color: #e5e7eb !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Parsers de salida
# -----------------------------

def _split_think_answer(text: str):
    import re
    if not isinstance(text, str):
        return None, ""

    # 1) Captura TODOS los bloques <think>…</think>
    thinks = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    reasoning = "\n\n---\n\n".join(t.strip() for t in thinks if t.strip()) or None

    # 2) Elimina TODOS los bloques <think>…</think> del texto
    remainder = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

    # 3) Limpia preámbulos de pensamiento comunes al inicio del texto
    #    (Thinking:, Reasoning:, Razonamiento:, Pensamiento:, Thought:, Chain of thought:)
    remainder = re.sub(
        r'^(?:\s*(?:thinking|reasoning|thoughts?|chain\s*of\s*thought|razonamiento|pensamiento)\s*[:：].*?(?:\n{2,}|$))+',
        "",
        remainder,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()

    return reasoning, remainder



def _as_final_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if "final_answer" in value and isinstance(value["final_answer"], str):
            return value["final_answer"]
        if "generate_final_answer" in value and isinstance(value["generate_final_answer"], str):
            return value["generate_final_answer"]
        return json.dumps(value, ensure_ascii=False)
    return str(value)





# -----------------------------
# Ejecución del grafo y streaming
# -----------------------------
def generate_response(
    user_input: str,
    source_mode: str,       # 'rag' | 'web' | 'hybrid'
    report_structure: str,
    max_search_queries: int,
    ollama_model: str,
    answer_length: str,     # 'short' | 'medium' | 'long'
    search_provider: str,   # 'tavily' | 'google' | 'serper' | 'ddg'
) -> str:
    initial_state = {"user_instructions": user_input}

    config = {"configurable": {
        "source_mode": source_mode,
        "report_structure": report_structure,
        "max_search_queries": max_search_queries,
        "ollama_model": ollama_model,
        "answer_length": answer_length,
        "search_provider": search_provider,
    }}

    final_raw = ""

    with st.status("**Investigador en ejecución...**", state="running") as langgraph_status:
        exp_queries = st.expander("Generar consultas de investigación", expanded=False)
        exp_web = st.expander("Búsqueda y resúmenes web", expanded=True)
        exp_final = st.expander("Generar respuesta final", expanded=False)

        for output in researcher.stream(initial_state, config=config):
            for key, value in output.items():
                if key == "generate_research_queries":
                    with exp_queries:
                        st.write(value)
                elif key.startswith("search_and_summarize_query"):
                    with exp_web:
                        with st.expander(key.replace("_", " ").title(), expanded=False):
                            st.write(value)
                elif key == "retrieve_rag_documents":
                    with exp_web:
                        with st.expander("Recuperación RAG", expanded=False):
                            st.write(value)
                elif key == "generate_final_answer":
                    final_raw = _as_final_text(value)
                    reasoning, answer = _split_think_answer(final_raw)
                    with exp_final:
                        if reasoning:
                            with st.expander("🧠 Razonamiento del modelo", expanded=False):
                                st.markdown(f"<div class='assistant-reasoning'>{reasoning}</div>", unsafe_allow_html=True)
                        with st.expander("📝 Respuesta generada", expanded=True):
                            st.markdown(f"<div class='assistant-answer'>{answer}</div>", unsafe_allow_html=True)

        langgraph_status.update(state="complete", label="**Usando LangGraph** (Investigación completada)")

    return final_raw or "No se generó ninguna respuesta"


# -----------------------------
# Utilidades de sesión
# -----------------------------
def clear_chat():
    st.session_state.messages = []
    st.session_state.processing_complete = False
    st.session_state.uploader_key = 0


# -----------------------------
# App
# -----------------------------
def main():
    st.set_page_config(page_title="🧠 Ollama WebRAG Researcher 🔍", layout="wide")
    inject_llmstudio_theme()

    # Estados
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_report_structure" not in st.session_state:
        st.session_state.selected_report_structure = None
    if "max_search_queries" not in st.session_state:
        st.session_state.max_search_queries = 5
    if "files_ready" not in st.session_state:
        st.session_state.files_ready = False
    if "ollama_model" not in st.session_state:
        st.session_state.ollama_model = None
    if "_cached_ollama_models" not in st.session_state:
        st.session_state._cached_ollama_models = get_installed_ollama_models()

    # Cabecera
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("🧠 Ollama WebRAG Researcher 🔍")
    with col2:
        if st.button("Limpiar chat", use_container_width=True):
            clear_chat()
            st.rerun()

    # Sidebar
    st.sidebar.title("🛠️ Ajustes")

    # Conmutador de fuente
    source_label = st.sidebar.radio(
        "Fuente de información",
        options=["Web + RAG", "Solo RAG", "Solo Web"],
        index=0,
        help="Elige si trabajar solo con tu base vectorial, solo con la web o combinar ambas.",
    )
    source_mode = {"Web + RAG": "hybrid", "Solo RAG": "rag", "Solo Web": "web"}[source_label]

    # Extensión de la respuesta
    length_label = st.sidebar.radio(
        "Extensión del informe",
        options=["Breve", "Media", "Amplia"],
        index=1,
        help="Controla lo conciso o detallado que quieres el resultado.",
    )
    answer_length = {"Breve": "short", "Media": "medium", "Amplia": "long"}[length_label]

    # Proveedor de búsqueda
    provider_label = st.sidebar.selectbox(
        "🔍 Proveedor de búsqueda web",
        options=["Tavily", "Google CSE", "Serper", "DuckDuckGo"],
        index=0,
        help="Requiere API keys para Google/Serper; DuckDuckGo no necesita clave.",
    )
    provider_map = {"Tavily": "tavily", "Google CSE": "google", "Serper": "serper", "DuckDuckGo": "ddg"}
    search_provider = provider_map[provider_label]

    # Estructuras de informe
    report_structures = get_report_structures()
    report_keys = list(report_structures.keys())
    try:
        default_index = list(map(str.lower, report_keys)).index("resumen_conciso")
    except ValueError:
        default_index = 0 if report_keys else 0
    selected_structure_key = st.sidebar.selectbox(
        "📈 Estructura del informe",
        options=report_keys,
        index=default_index if report_keys else 0,
    )
    if report_keys:
        st.session_state.selected_report_structure = report_structures[selected_structure_key]

    # Nº máximo de búsquedas web
    st.session_state.max_search_queries = st.sidebar.number_input(
        "🌐 Nº máximo de búsquedas web",
        min_value=1,
        max_value=30,
        value=st.session_state.max_search_queries,
        help="Establece el número máximo de búsquedas web (1-30).",
    )

    # Selector de modelo de Ollama
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠  Modelo de Ollama")
    if st.sidebar.button("🔄 Refrescar modelos"):
        st.session_state._cached_ollama_models = get_installed_ollama_models()
    models = st.session_state._cached_ollama_models
    if models:
        default_model_index = models.index(st.session_state.ollama_model) if st.session_state.ollama_model in models else 0
        selected_model = st.sidebar.selectbox("Selecciona el modelo para responder", options=models, index=default_model_index)
        st.session_state.ollama_model = selected_model
    else:
        st.sidebar.error(
            "No se han encontrado modelos en Ollama o el servidor no está accesible.\n\n"
            f"Intenté: {_ollama_base_url()}/api/tags"
        )

    st.sidebar.markdown("---")

    # Subida de archivos
    uploaded_files = st.sidebar.file_uploader(
        "📑 Subir nuevos documentos",
        type=["pdf", "txt", "csv", "md"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )
    if uploaded_files:
        st.session_state.files_ready = True
        st.session_state.processing_complete = False

    if st.session_state.files_ready and not st.session_state.processing_complete:
        process_button_placeholder = st.sidebar.empty()
        with process_button_placeholder.container():
            process_clicked = st.button("Procesar archivos", use_container_width=True)
        if process_clicked:
            with process_button_placeholder:
                with st.status("Procesando archivos...", expanded=False) as status:
                    if process_uploaded_files(uploaded_files):
                        st.session_state.processing_complete = True
                        st.session_state.files_ready = False
                        st.session_state.uploader_key += 1
                    status.update(label="¡Archivos procesados correctamente!", state="complete", expanded=False)

    # Historial de chat
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                reasoning, answer = _split_think_answer(message["content"])
                if reasoning:
                    with st.expander("🧠 Razonamiento del modelo", expanded=False):
                        st.markdown(f"<div class='assistant-reasoning'>{reasoning}</div>", unsafe_allow_html=True)
                with st.expander("📝 Respuesta generada", expanded=True):
                    st.markdown(f"<div class='assistant-answer'>{answer}</div>", unsafe_allow_html=True)
                if st.button("📋", key=f"copy_hist_{idx}"):
                    pyperclip.copy(answer)
            else:
                st.write(message["content"])

    # Entrada del chat
    if user_input := st.chat_input("Escribe tu mensaje aquí..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        report_structure = (
            st.session_state.selected_report_structure["content"]
            if st.session_state.selected_report_structure
            else ""
        )
        final_raw = generate_response(
            user_input=user_input,
            source_mode=source_mode,
            report_structure=report_structure,
            max_search_queries=st.session_state.max_search_queries,
            ollama_model=st.session_state.ollama_model or "qwen3:14b",
            answer_length=answer_length,
            search_provider=search_provider,
        )

        st.session_state.messages.append({"role": "assistant", "content": final_raw})
        last_idx = len(st.session_state.messages) - 1
        with st.chat_message("assistant"):
            reasoning, answer = _split_think_answer(final_raw)
            if reasoning:
                with st.expander("🧠 Razonamiento del modelo", expanded=False):
                    st.markdown(f"<div class='assistant-reasoning'>{reasoning}</div>", unsafe_allow_html=True)
            with st.expander("📝 Respuesta generada", expanded=True):
                st.markdown(f"<div class='assistant-answer'>{answer}</div>", unsafe_allow_html=True)
            if st.button("📋", key=f"copy_new_{last_idx}"):
                pyperclip.copy(answer)


if __name__ == "__main__":
    main()
