# Cliente para hablar con el LLM que genera los itinerarios.
# Soporta dos proveedores (elegido en config.PROVEEDOR):
#   - "ollama": modelo corriendo en tu propio ordenador, gratis, vía API REST local.
#   - "groq": modelo en la nube (API compatible con OpenAI), gratis con API key, para poder
#             desplegar la app sin depender de que tu PC esté encendida.

import requests
from config import PROVEEDOR, MODELO_OLLAMA, MODELO_GROQ, GROQ_API_KEY, OLLAMA_URL, GROQ_URL


def generar_texto(prompt, system=None):
    """Le pide al LLM que genere texto. Usa Ollama o Groq según config.PROVEEDOR."""
    if PROVEEDOR == "groq":
        return _generar_texto_groq(prompt, system)
    return _generar_texto_ollama(prompt, system)


def _generar_texto_ollama(prompt, system=None):
    payload = {
        "model": MODELO_OLLAMA,
        "prompt": prompt,
        "system": system,
        "stream": False,
    }
    print(f"[llm_client] Llamando a Ollama ({MODELO_OLLAMA})...")
    respuesta = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=300)
    respuesta.raise_for_status()
    texto = respuesta.json()["response"]
    print(f"[llm_client] Respuesta recibida ({len(texto)} caracteres)")
    return texto


def _generar_texto_groq(prompt, system=None):
    if not GROQ_API_KEY:
        raise RuntimeError(
            "PROVEEDOR=groq pero no hay GROQ_API_KEY configurada. "
            "Consigue una gratis en console.groq.com y ponla como variable de entorno o secreto."
        )
    mensajes = []
    if system:
        mensajes.append({"role": "system", "content": system})
    mensajes.append({"role": "user", "content": prompt})

    payload = {"model": MODELO_GROQ, "messages": mensajes}
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    print(f"[llm_client] Llamando a Groq ({MODELO_GROQ})...")
    respuesta = requests.post(GROQ_URL, json=payload, headers=headers, timeout=120)
    respuesta.raise_for_status()
    texto = respuesta.json()["choices"][0]["message"]["content"]
    print(f"[llm_client] Respuesta recibida ({len(texto)} caracteres)")
    return texto
