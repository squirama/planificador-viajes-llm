# Control de calidad de los itinerarios generados (requisito 7 del enunciado).
# Combina comprobaciones automáticas simples (rápidas, sin LLM) con una auto-revisión
# hecha por el propio modelo (más lenta, pero detecta cosas que el código no puede ver).

import re
from llm_client import generar_texto
from prompts import construir_prompt_autocritica


def comprobaciones_basicas(preferencias, itinerario):
    """
    Comprobaciones de código normal y corriente (sin LLM), rápidas y fiables.
    Devuelve una lista de avisos (vacía si todo está bien).
    """
    avisos = []

    if not itinerario or len(itinerario.strip()) < 20:
        avisos.append("El itinerario generado está vacío o es demasiado corto.")
        return avisos

    dias_encontrados = len(re.findall(r"[Dd]ía\s*\d+", itinerario))
    dias_pedidos = preferencias["duracion"]
    if dias_encontrados != dias_pedidos:
        avisos.append(
            f"Se pidieron {dias_pedidos} días pero el itinerario menciona {dias_encontrados}."
        )

    if preferencias["destino"].lower() not in itinerario.lower():
        avisos.append("El itinerario no menciona el destino solicitado.")

    return avisos


def autocritica_llm(preferencias, itinerario):
    """
    Le pedimos al propio LLM que revise su respuesta y señale problemas
    (sesgos, incoherencias, incumplimientos). Devuelve una lista de avisos.
    """
    prompt = construir_prompt_autocritica(preferencias, itinerario)
    respuesta = generar_texto(prompt)

    if respuesta.strip().upper().startswith("OK"):
        return []
    # Cada línea de la respuesta la tratamos como un aviso
    return [linea.strip("- ").strip() for linea in respuesta.split("\n") if linea.strip()]


def revisar_itinerario(preferencias, itinerario):
    """Junta las dos comprobaciones en una sola lista de avisos."""
    avisos = comprobaciones_basicas(preferencias, itinerario)
    avisos += autocritica_llm(preferencias, itinerario)
    return avisos
