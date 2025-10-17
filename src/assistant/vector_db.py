"""
Autor: Luis González Fraga

Asunciones:
- Requiere Python >= 3.13
"""
import os
from typing import List

from langchain_community.document_loaders import DirectoryLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings  # <— NUEVO import (corrige la deprecación)
from langchain_chroma import Chroma
from langchain_core.documents import Document


# =========================
# Configuración de Embeddings (Ollama)
# =========================
# Puedes cambiar estos valores por variables de entorno si lo prefieres.
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large:latest")
OLLAMA_BASE_URL = (
    os.getenv("OLLAMA_BASE_URL")
    or os.getenv("OLLAMA_HOST")
    or "http://127.0.0.1:11434"
)

# Carpeta de persistencia de Chroma
VECTOR_DB_PATH = "database"


def _get_embeddings() -> OllamaEmbeddings:
    """
    Devuelve una instancia de OllamaEmbeddings con el modelo local configurado.
    Lanza un error legible si Ollama no está accesible.
    """
    try:
        return OllamaEmbeddings(
            model=OLLAMA_EMBED_MODEL,
            base_url=OLLAMA_BASE_URL,
        )
    except Exception as e:
        raise RuntimeError(
            f"Error creando OllamaEmbeddings con modelo '{OLLAMA_EMBED_MODEL}' "
            f"en '{OLLAMA_BASE_URL}'. Asegúrate de que Ollama está en ejecución "
            f"(p. ej., 'ollama serve') y que el modelo existe. Detalle: {e}"
        ) from e


def _semantic_split(
    documents: List[Document],
    embeddings: OllamaEmbeddings,
    chunk_size: int = 2000,
    chunk_overlap: int = 400,
) -> List[Document]:
    """
    Aplica un pipeline de troceado:
    1) SemanticChunker (basado en embeddings) para cortes semánticos.
    2) RecursiveCharacterTextSplitter para garantizar tamaños controlados.

    Si SemanticChunker falla por cualquier motivo, hace fallback directo
    a RecursiveCharacterTextSplitter.
    """
    try:
        # 1) Cortado semántico (más coherente)
        semantic_text_splitter = SemanticChunker(embeddings)
        sem_docs = semantic_text_splitter.split_documents(documents)
    except Exception:
        # Fallback si SemanticChunker no está disponible o falla
        sem_docs = documents

    # 2) Límite de tamaño con RCTS
    rcts = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return rcts.split_documents(sem_docs)


def get_or_create_vector_db():
    """
    Obtiene la base vectorial Chroma si existe; si no, la crea cargando documentos
    desde ./files y calculando embeddings con Ollama.
    """
    embeddings = _get_embeddings()

    if os.path.exists(VECTOR_DB_PATH) and os.listdir(VECTOR_DB_PATH):
        # Usa el almacén existente
        vectorstore = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=embeddings
        )
    else:
        # Carga documentos desde ./files
        loader = DirectoryLoader("./files")
        docs = loader.load()

        # Troceado (semántico + tamaño)
        split_documents = _semantic_split(docs, embeddings)

        # Crea y persiste la base
        vectorstore = Chroma.from_documents(
            split_documents,
            embeddings,
            persist_directory=VECTOR_DB_PATH
        )

    return vectorstore


def add_documents(documents: List[Document]):
    """
    Añade nuevos documentos al vector store existente (o crea uno si no existe).
    Trocea los documentos (semántico + tamaño) y persiste.
    """
    embeddings = _get_embeddings()

    # Troceado (semántico + tamaño)
    split_documents = _semantic_split(documents, embeddings)

    if os.path.exists(VECTOR_DB_PATH) and os.listdir(VECTOR_DB_PATH):
        # Añade al vector store existente
        vectorstore = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=embeddings
        )
        vectorstore.add_documents(split_documents)
    else:
        # Crea un nuevo vector store si no existe
        vectorstore = Chroma.from_documents(
            split_documents,
            embeddings,
            persist_directory=VECTOR_DB_PATH
        )

    return vectorstore
