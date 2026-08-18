import os


def _leer_config(nombre, por_defecto=""):
    """
    Lee un valor de configuración de dos sitios posibles:
    1. Los "Secrets" de Streamlit Cloud (cuando la app está desplegada ahí).
    2. Una variable de entorno normal (cuando corres la app en tu propio ordenador).
    """
    try:
        import streamlit as st
        if nombre in st.secrets:
            return st.secrets[nombre]
    except Exception:
        pass
    return os.environ.get(nombre, por_defecto)


# Proveedor del LLM de generación: "ollama" (local, gratis, necesita tu PC encendido)
# o "groq" (nube, gratis con API key, sirve para desplegar la app con una URL pública).
PROVEEDOR = _leer_config("PROVEEDOR", "ollama")

MODELO_OLLAMA = "llama3.1:latest"           # modelo local (ya descargado con `ollama pull`)
MODELO_GROQ = "openai/gpt-oss-120b"         # modelo servido por Groq (gratis, en la nube)
GROQ_API_KEY = _leer_config("GROQ_API_KEY", "")

OLLAMA_URL = "http://localhost:11434"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Modelo de embeddings para el RAG. Corre en local (CPU) con sentence-transformers,
# así que funciona igual tanto si generas texto con Ollama como con Groq.
MODELO_EMBEDDINGS = "all-MiniLM-L6-v2"

CARPETA_GUIAS = "data/guides"

NUM_FRAGMENTOS_RAG = 4  # cuántos fragmentos de contexto recuperamos para cada consulta
