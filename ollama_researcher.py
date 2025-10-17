"""
Autor: Luis González Fraga

Asunciones:
- Requiere Python >= 3.13
"""
import os
import sys
from dotenv import load_dotenv

from src.assistant.graph import researcher
from src.assistant.vector_db import get_or_create_vector_db

load_dotenv()

# Toma el tema desde CLI o env
topic = " ".join(sys.argv[1:]).strip() or os.environ.get("RESEARCH_TOPIC", "").strip()
if not topic:
    topic = "Escribe aquí tu tema de investigación"

# PLANTILLA básica (puedes sobreescribirla leyendo un archivo)
report_structure = """
1. Introducción
- Alcance y objetivo.

2. Desarrollo
- Secciones temáticas con hallazgos y detalles.

3. Puntos clave
- Viñetas con ideas principales.

4. Conclusión
- Síntesis y relevancia.
"""

# Modo y proveedor (por ENV)
source_mode = os.environ.get("SOURCE_MODE", "hybrid").strip().lower()     # 'rag'|'web'|'hybrid'
web_provider = os.environ.get("WEB_PROVIDER", "tavily").strip().lower()   # 'tavily'|'google'|'serper'|'duckduckgo'
max_q = int(os.environ.get("MAX_SEARCH_QUERIES", "5"))
ollama_model = os.environ.get("OLLAMA_MODEL", "qwen3:14b")
answer_length = os.environ.get("ANSWER_LENGTH", "medium")                 # 'short'|'medium'|'long'

# Estado inicial
initial_state = {"user_instructions": topic}

# Configuración
config = {
    "configurable": {
        "source_mode": source_mode,
        "web_provider": web_provider,
        "report_structure": report_structure,
        "max_search_queries": max_q,
        "ollama_model": ollama_model,
        "answer_length": answer_length,
    }
}

# Inicializa la vector DB (para modos 'rag' o 'hybrid')
_ = get_or_create_vector_db()

# Ejecuta el grafo
for output in researcher.stream(initial_state, config=config):
    for key, value in output.items():
        print(f"\n=== Finished: {key} ===")
        print(value)
