# Aquí construimos los prompts (las instrucciones que le mandamos al LLM).
# Esto es la técnica de "prompting" que pide el requisito 3 del enunciado.

SYSTEM_PROMPT = """Eres un planificador de viajes experto con años de experiencia creando itinerarios
personalizados. Tu trabajo es crear itinerarios claros, realistas y atractivos, respetando siempre
el presupuesto, la duración y los intereses del usuario. Organiza siempre la respuesta día por día,
usando el formato "Día 1:", "Día 2:", etc. No inventes datos peligrosos ni contenido inapropiado."""


def construir_prompt_itinerario(preferencias, contexto_rag, itinerario_anterior=None, feedback=None):
    """
    Junta las preferencias del usuario + el contexto recuperado por RAG en un solo prompt.
    Si hay un itinerario anterior y feedback del usuario, se los incluimos para que el modelo
    lo corrija en vez de empezar de cero (esto es el bucle de refinamiento del requisito 6).
    """
    partes = []

    partes.append(f"Destino: {preferencias['destino']}")
    partes.append(f"Duración: {preferencias['duracion']} días")
    partes.append(f"Presupuesto total aproximado: {preferencias['presupuesto']} euros")
    partes.append(f"Intereses: {', '.join(preferencias['intereses'])}")
    if preferencias.get("restricciones"):
        partes.append(f"Restricciones o requisitos especiales: {preferencias['restricciones']}")

    if contexto_rag:
        partes.append("\nInformación de referencia sobre el destino (úsala si es relevante):")
        partes.append(contexto_rag)

    if itinerario_anterior and feedback:
        partes.append(f"\nItinerario generado anteriormente:\n{itinerario_anterior}")
        partes.append(f"\nEl usuario ha pedido este cambio: {feedback}")
        partes.append("\nGenera una versión nueva del itinerario aplicando ese cambio.")
    else:
        partes.append(f"\nGenera un itinerario de viaje día por día para {preferencias['duracion']} días.")

    return "\n".join(partes)


def construir_prompt_autocritica(preferencias, itinerario):
    """
    Prompt para que el propio LLM revise su itinerario (auto-crítica, requisito 7).
    """
    return f"""Revisa este itinerario de viaje y comprueba si cumple lo pedido por el usuario.

Preferencias del usuario:
- Destino: {preferencias['destino']}
- Duración: {preferencias['duracion']} días
- Presupuesto: {preferencias['presupuesto']} euros
- Intereses: {', '.join(preferencias['intereses'])}

Itinerario a revisar:
{itinerario}

Responde SOLO con una lista corta de problemas encontrados (por ejemplo: número de días incorrecto,
actividades que no encajan con los intereses, contenido sesgado o inapropiado, precios poco realistas).
Si no encuentras ningún problema, responde exactamente: OK"""
