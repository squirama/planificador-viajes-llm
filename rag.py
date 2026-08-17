# RAG = Retrieval-Augmented Generation (Generación Aumentada por Recuperación).
# Aquí tenemos una mini "base de conocimiento" de fichas de destinos (carpeta data/guides).
# Antes de generar el itinerario, buscamos qué fragmentos de esas fichas son más relevantes
# para la consulta del usuario y se los pasamos al LLM como contexto extra.
#
# Los embeddings (vectores que representan el significado del texto) se calculan en local
# con sentence-transformers, corriendo en CPU. Así funciona igual de bien tanto si el LLM de
# generación es Ollama (local) como Groq (nube), porque no depende de ninguno de los dos.

import os

# sentence-transformers intenta cargar TensorFlow si lo encuentra instalado, y en algunos
# equipos esa instalación de TensorFlow está rota o desactualizada. Como aquí solo usamos
# PyTorch, le decimos que ni lo intente (hay que ponerlo ANTES de importar sentence_transformers).
os.environ.setdefault("USE_TF", "0")

import numpy as np
from sentence_transformers import SentenceTransformer
from config import CARPETA_GUIAS, NUM_FRAGMENTOS_RAG, MODELO_EMBEDDINGS

_modelo_embeddings = None  # se carga una sola vez y se reutiliza (ver obtener_modelo_embeddings)


def obtener_modelo_embeddings():
    """Carga el modelo de embeddings la primera vez que se necesita (es lento cargarlo)."""
    global _modelo_embeddings
    if _modelo_embeddings is None:
        print(f"[rag] Cargando modelo de embeddings {MODELO_EMBEDDINGS}...")
        _modelo_embeddings = SentenceTransformer(MODELO_EMBEDDINGS)
    return _modelo_embeddings


def cargar_fragmentos():
    """
    Lee todos los ficheros .md de data/guides y los trocea en fragmentos (por párrafo).
    Devuelve una lista de textos.
    """
    fragmentos = []
    for nombre_archivo in os.listdir(CARPETA_GUIAS):
        if not nombre_archivo.endswith(".md"):
            continue
        ruta = os.path.join(CARPETA_GUIAS, nombre_archivo)
        with open(ruta, "r", encoding="utf-8") as f:
            contenido = f.read()
        # Troceamos por párrafos (líneas separadas por doble salto de línea)
        for parrafo in contenido.split("\n\n"):
            parrafo = parrafo.strip()
            if len(parrafo) > 20:  # ignoramos fragmentos demasiado cortos (títulos sueltos, etc.)
                fragmentos.append(parrafo)
    print(f"[rag] Cargados {len(fragmentos)} fragmentos desde {CARPETA_GUIAS}")
    return fragmentos


def calcular_embeddings(fragmentos):
    """
    Calcula el embedding de todos los fragmentos de golpe (mucho más rápido que uno a uno).
    En app.py se cachea con st.cache_resource para no repetir este cálculo en cada interacción.
    """
    print(f"[rag] Calculando embeddings de {len(fragmentos)} fragmentos...")
    modelo = obtener_modelo_embeddings()
    return modelo.encode(fragmentos)


def similitud_coseno(a, b):
    """Mide qué tan parecidos son dos vectores (1 = iguales, 0 = nada que ver)."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def buscar_contexto(consulta, fragmentos, embeddings_fragmentos):
    """
    Dado un texto de consulta (ej: destino + intereses), busca los fragmentos más
    parecidos en la base de conocimiento y los devuelve unidos en un solo texto.
    """
    modelo = obtener_modelo_embeddings()
    vector_consulta = modelo.encode([consulta])[0]
    puntuaciones = [similitud_coseno(vector_consulta, v) for v in embeddings_fragmentos]

    # Cogemos los índices de los fragmentos con mayor puntuación
    mejores_indices = np.argsort(puntuaciones)[::-1][:NUM_FRAGMENTOS_RAG]

    seleccionados = [fragmentos[i] for i in mejores_indices]
    print(f"[rag] Fragmentos seleccionados para la consulta: {len(seleccionados)}")
    return "\n\n".join(seleccionados)
