import os

# Proveedor del LLM de generación: "ollama" (local, gratis, necesita tu PC encendido)
# o "groq" (nube, gratis con API key, sirve para desplegar la app con una URL pública).
# Se elige con la variable de entorno PROVEEDOR; si no se define, se usa Ollama por defecto.
PROVEEDOR = os.environ.get("PROVEEDOR", "ollama")

MODELO_OLLAMA = "llama3.1:latest"           # modelo local (ya descargado con `ollama pull`)
MODELO_GROQ = "llama-3.3-70b-versatile"     # modelo servido por Groq (gratis, en la nube)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

OLLAMA_URL = "http://localhost:11434"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Modelo de embeddings para el RAG. Corre en local (CPU) con sentence-transformers,
# así que funciona igual tanto si generas texto con Ollama como con Groq.
MODELO_EMBEDDINGS = "all-MiniLM-L6-v2"

CARPETA_GUIAS = "data/guides"

NUM_FRAGMENTOS_RAG = 4  # cuántos fragmentos de contexto recuperamos para cada consulta
